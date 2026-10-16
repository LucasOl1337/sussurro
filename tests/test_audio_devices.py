"""Friendly microphone labels do not replace stable PortAudio identifiers."""

import unittest

from sussurro_audio import describe_input, pipewire_inputs, prepare_input_devices, source_label


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



PORTAUDIO = [{"name": n, "index": i, "max_input_channels": 2} for i, n in enumerate(
    ["HD-Audio Generic: ALC897 Analog (hw:3,0)", "FIFINE Microphone: USB Audio (hw:4,0)",
     "lavrate", "samplerate", "speex", "upmix", "jack", "pipewire", "pulse", "default"])]
SOURCES = [
    {"name": "alsa_output.usb-MCHOSE_X9-01.analog-stereo.monitor",
     "description": "Monitor of MCHOSE X9 Analog Stereo", "properties": {"device.class": "monitor"}},
    {"name": "alsa_input.usb-MCHOSE_X9-01.mono-fallback", "description": "MCHOSE X9 Mono",
     "properties": {"device.class": "sound"}},
    {"name": "alsa_input.usb-3142_FIFINE_Microphone-00.mono-fallback",
     "description": "FIFINE Microphone Mono", "properties": {"device.class": "sound"}},
    {"name": "dino_prova.monitor", "description": "Monitor of dino_prova", "properties": {}},
]


class PipeWireInputTests(unittest.TestCase):
    def test_only_system_default_and_physical_microphones(self):
        result = pipewire_inputs(PORTAUDIO, SOURCES, "alsa_input.usb-MCHOSE_X9-01.mono-fallback")
        self.assertEqual([d.label for d in result.values()],
                         ["Padrão do PC", "MCHOSE X9", "FIFINE Microphone"])
        default = result["pipewire"]
        self.assertEqual((default.index, default.source), (7, None))
        self.assertIn("Agora: MCHOSE X9", default.description)
        fifine = result["pulse:alsa_input.usb-3142_FIFINE_Microphone-00.mono-fallback"]
        self.assertEqual(fifine.index, 8)  # grava pela rota pulse com PULSE_SOURCE
        self.assertEqual(fifine.source, "alsa_input.usb-3142_FIFINE_Microphone-00.mono-fallback")

    def test_without_pipewire_route_caller_falls_back(self):
        self.assertEqual(pipewire_inputs(PORTAUDIO[:7], SOURCES), {})

    def test_channel_suffix_is_dropped(self):
        self.assertEqual(source_label("MCHOSE X9 Mono"), "MCHOSE X9")
        self.assertEqual(source_label("Webcam Analog Stereo"), "Webcam")


if __name__ == "__main__":
    unittest.main()
