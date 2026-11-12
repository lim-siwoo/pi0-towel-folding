"""Offline regression checks; no robot modules or hardware are loaded."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from record_towel import cleanup_devices, run_recording


def fake_arm(connected=True):
    arm = SimpleNamespace(driver=Mock(), disconnect=Mock())
    arm.driver.get_is_configured.return_value = connected
    return arm


class RecordingTests(unittest.TestCase):
    def test_partial_group_still_returns_each_connected_arm_home(self):
        left, right = fake_arm(), fake_arm(False)
        camera = SimpleNamespace(is_connected=True, disconnect=Mock())
        device = SimpleNamespace(left_arm=left, right_arm=right, cameras={"high": camera}, is_connected=False)
        cleanup_devices([device], "position")
        left.driver.set_all_modes.assert_called_once_with("position")
        left.disconnect.assert_called_once()
        right.disconnect.assert_not_called()
        camera.disconnect.assert_called_once()

    def test_one_home_failure_does_not_skip_other_arms(self):
        left, right = fake_arm(), fake_arm()
        left.disconnect.side_effect = RuntimeError("link failed")
        with self.assertLogs(level="ERROR"):
            cleanup_devices([SimpleNamespace(left_arm=left, right_arm=right)], "position")
        left.driver.cleanup.assert_called_once()
        right.disconnect.assert_called_once()

    def test_saved_action_is_sent_target_and_cleanup_survives_save_failure(self):
        robot = SimpleNamespace(send_action=Mock(return_value={"joint.pos": 0.05}),
                                left_arm=fake_arm(), right_arm=fake_arm())
        teleop = SimpleNamespace(left_arm=fake_arm(), right_arm=fake_arm())
        original_frame = Mock(side_effect=lambda features, values, prefix: dict(values))
        module = SimpleNamespace(make_robot_from_config=Mock(return_value=robot),
                                 make_teleoperator_from_config=Mock(return_value=teleop),
                                 build_dataset_frame=original_frame)
        factory = module.make_robot_from_config

        def failing_record(cfg):
            r = module.make_robot_from_config(None)
            module.make_teleoperator_from_config(None)
            r.send_action({"joint.pos": 0.9})
            self.assertEqual(module.build_dataset_frame({}, {"joint.pos": 0.9}, prefix="action"),
                             {"joint.pos": 0.05})
            self.assertEqual(module.build_dataset_frame({}, {"joint.pos": 0.7}, prefix="observation"),
                             {"joint.pos": 0.7})
            raise OSError("dataset finalization failed")

        module.record = failing_record
        with self.assertRaisesRegex(OSError, "finalization failed"):
            run_recording(module, None, "position")
        for device in (robot, teleop):
            device.left_arm.disconnect.assert_called_once()
            device.right_arm.disconnect.assert_called_once()
        self.assertIs(module.make_robot_from_config, factory)
        self.assertIs(module.build_dataset_frame, original_frame)


if __name__ == "__main__":
    unittest.main()
