from typing import Dict, Any

from dataset.p14t import Phoenix14T


class CSLDaily(Phoenix14T):
    """
    Dataset class for the CSL-Daily Chinese sign language dataset.

    Uses the same annotation and feature layout as Phoenix14T
    (see scripts/prepare_csl_daily.py) and only differs in the target
    language, the text normalization and the video path convention.
    """
    def __getitem__(self, index: int) -> Dict[str, Any]:
        result = super().__getitem__(index)
        data = self.data[index]

        # Chinese sentences end with full-width punctuation (。？！), so keep
        # the text as is instead of appending a '.'
        result['text'] = data['text'].strip()
        result['lang'] = 'Chinese'
        result['vid_path'] = str(self.vid_root / 'CSL-Daily_256x256px' / data['folder'])

        return result
