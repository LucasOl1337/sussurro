"""Execution choices distinguish runnable backends from merely detected hardware."""

import unittest

from sussurro_hardware import HardwareInfo, execution_device_labels, execution_hardware_note


class HardwarePresentationTests(unittest.TestCase):
    def test_labels_name_nvidia_and_cpu(self):
        hardware = HardwareInfo("AMD Ryzen 7 9800X3D 8-Core Processor",
                                ("GeForce RTX 4070 Ti SUPER",), ("Radeon Graphics",))
        labels = execution_device_labels(hardware)
        self.assertEqual(set(labels), {"auto", "cuda", "cpu"})
        self.assertIn("RTX 4070 Ti SUPER", labels["cuda"])
        self.assertIn("Ryzen 7 9800X3D", labels["cpu"])

    def test_amd_is_explained_but_not_offered_as_a_fake_backend(self):
        hardware = HardwareInfo(amd_gpus=("Radeon Graphics",))
        labels = execution_device_labels(hardware)
        self.assertNotIn("amd", labels)
        self.assertIn("CPU", labels["auto"])
        note = execution_hardware_note(hardware)
        self.assertIn("Radeon Graphics", note)
        self.assertIn("ROCm", note)
        self.assertIn("não se soma", note)


if __name__ == "__main__":
    unittest.main()
