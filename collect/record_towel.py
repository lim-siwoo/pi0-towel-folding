"""User-operated LeRobot recorder. --check only parses config; no hardware is opened."""
import argparse
import fcntl
import json
import logging
import os
from pathlib import Path
import signal
import sys

HERE = Path(__file__).resolve().parent
DATASET_ROOT = HERE.parent / "datasets/recorded/towel_new_fold_20261106"


def cleanup_devices(devices, position_mode):
    """Handle partially connected groups and keep trying the other arms on failure."""
    for device in devices:
        for name in ("left_arm", "right_arm"):
            arm = getattr(device, name, None)
            if arm is None:
                continue
            try:
                if not arm.driver.get_is_configured():
                    continue
                arm.driver.set_all_modes(position_mode)
                # Installed Trossen disconnect returns staged -> all-zero rest.
                arm.disconnect()
            except BaseException:
                logging.exception("%s home/cleanup failed; continuing other devices", name)
                try:
                    arm.driver.cleanup()
                except BaseException:
                    logging.exception("%s driver cleanup failed", name)
        for camera in getattr(device, "cameras", {}).values():
            try:
                if camera.is_connected:
                    camera.disconnect()
            except BaseException:
                logging.exception("Camera cleanup failed")


def run_recording(module, cfg, position_mode):
    """Use upstream recording, retaining the sent action and a final cleanup fallback."""
    devices = []
    last_sent = None
    old_robot = module.make_robot_from_config
    old_teleop = module.make_teleoperator_from_config
    old_frame = module.build_dataset_frame

    def make_robot(config):
        robot = old_robot(config)
        devices.append(robot)
        send = robot.send_action

        def send_and_remember(action):
            nonlocal last_sent
            sent = send(action)
            last_sent = dict(sent)
            return sent

        robot.send_action = send_and_remember
        return robot

    def make_teleop(config):
        teleop = old_teleop(config)
        devices.append(teleop)
        return teleop

    def build_frame(features, values, prefix):
        # Installed LeRobot saves leader targets even when the follower clips them.
        # Save actual follower targets instead, using exactly the same feature order.
        if prefix == "action":
            if last_sent is None:
                raise RuntimeError("No successfully sent action is available for this frame")
            values = last_sent
        return old_frame(features, values, prefix=prefix)

    module.make_robot_from_config = make_robot
    module.make_teleoperator_from_config = make_teleop
    module.build_dataset_frame = build_frame
    try:
        return module.record(cfg)
    finally:
        try:
            cleanup_devices(devices, position_mode)
        finally:
            module.make_robot_from_config = old_robot
            module.make_teleoperator_from_config = old_teleop
            module.build_dataset_frame = old_frame


def active_robot_clients():
    result = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            args = (entry / "cmdline").read_bytes().decode(errors="replace").split("\0")
        except OSError:
            continue
        if not args or not Path(args[0]).name.startswith("python"):
            continue
        if any(arg.endswith("/trossen_aloha_real/main.py")
               or arg.endswith("/base_model_run/main.py") for arg in args[1:]):
            result.append(entry.name)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate settings without hardware")
    options, overrides = parser.parse_known_args()
    import draccus
    from lerobot.scripts import lerobot_record as recorder
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()
    cfg = draccus.parse(recorder.RecordConfig, config_path=str(HERE / "new_fold_20261106.json"), args=overrides)
    if cfg.dataset.single_task != "fold towel":
        raise ValueError("This collection uses exactly: fold towel")
    if Path(cfg.dataset.root).resolve() != DATASET_ROOT.resolve():
        raise ValueError("Use the new collection root; existing datasets must stay separate")
    if cfg.dataset.push_to_hub:
        raise ValueError("This collection is local only")
    for key in ("fps", "num_episodes", "episode_time_s", "reset_time_s"):
        if getattr(cfg.dataset, key) <= 0:
            raise ValueError(f"dataset.{key} must be positive")
    if cfg.resume:
        tasks = DATASET_ROOT / "meta/tasks.parquet"
        if not tasks.is_file():
            raise ValueError("No existing dataset to resume; omit --resume=true on the first run")
        import pandas as pd
        if set(pd.read_parquet(tasks).index) != {"fold towel"}:
            raise ValueError("Existing task text differs from fold towel")
    elif DATASET_ROOT.exists():
        raise ValueError("Collection already exists. To append episodes, use --resume=true")
    print(json.dumps({"task": cfg.dataset.single_task, "root": str(cfg.dataset.root),
                      "fps": cfg.dataset.fps, "episodes": cfg.dataset.num_episodes,
                      "episode_time_s": cfg.dataset.episode_time_s,
                      "reset_time_s": cfg.dataset.reset_time_s,
                      "cameras": list(cfg.robot.cameras), "resume": cfg.resume,
                      "hardware_opened": False}, ensure_ascii=False, indent=2), flush=True)
    if options.check:
        return
    if not sys.stdin.isatty():
        raise RuntimeError("Run this command yourself in an interactive terminal")
    clients = active_robot_clients()
    if clients:
        raise RuntimeError(f"Robot policy client still running (PID {', '.join(clients)}); finish it first")
    # Match the existing collection tool's lock, without starting its service.
    with Path("/home/dit/control_center/data/robot.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        import trossen_arm

        def stop(signum, frame):
            raise KeyboardInterrupt("Recording interrupted; returning connected arms home")

        previous = signal.signal(signal.SIGTERM, stop)
        try:
            run_recording(recorder, cfg, trossen_arm.Mode.position)
        finally:
            signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
