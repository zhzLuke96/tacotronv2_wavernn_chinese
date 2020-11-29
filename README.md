# TacotronV2 + WaveRNN

```python
# TacotronV2
python tacotron_synthesize.py --text '现在是凌晨零点二十七分，帮您订好上午八点的闹钟。'

# WaveRNN
python wavernn_gen.py --file path_to_mel_generated_by_tacotronv2 
```

tacotron.datasets
tacotron.utils.infolog
tacotron.models
tacotron.utils
tacotron.utils.text
tacotron.pinyin.parse_text_to_pyin

wavernn.utils.dataset
wavernn.utils.dsp
wavernn.models.fatchord_version
wavernn.utils.paths
wavernn.utils.display

conda activate py36