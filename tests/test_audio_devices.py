"""Friendly microphone labels do not replace stable PortAudio identifiers."""

import unittest

from sussurro_audio import describe_input, prepare_input_devices


class AudioDevicePresentationTests(unittest.TestCase):
    def test_pipewire_explains_and_names_current_system_microphone(self):
        device = describe_input("pipewire", 21, "FIFINE Microphone")
        self.assertEqual(device.name, "pipewire")
        self.assertIn("FIFINE Microphone", device.label)
        self.assertIn("PipeWire", device.label)
        self.assertIn("Segue automaticamente", device.description)

    def test_direct_usb_device_gets_human_name_and_explanation(self):
        device = describe_input("FIFINE Microphone: USB Audio (hw:4,0)", 11)
        self.assertEqual(device.label, "FIFINE Microphone — USB direto")
        self.assertIn("diretamente", device.description)

    def test_system_routes_are_first_and_duplicate_labels_are_unique(self):
        devices = [
            {"name": "Mic: USB Audio (hw:1,0)", "index": 4, "max_input_channels": 1},
            {"name": "pipewire", "index": 2, "max_input_channels": 128},
            {"name": "Mic: USB Audio (hw:2,0)", "index": 5, "max_input_channels": 1},
            {"name": "speaker", "index": 9, "max_input_channels": 0},
        ]
        result = prepare_input_devices(devices, "Mic")
        self.assertEqual(list(result)[0], "pipewire")
        self.assertNotEqual(result["Mic: USB Audio (hw:1,0)"].label,
                            result["Mic: USB Audio (hw:2,0)"].label)


if __name__ == "__main__":
    unittest.main()
