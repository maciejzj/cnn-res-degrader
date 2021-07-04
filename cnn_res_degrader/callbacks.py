from pathlib import Path

from tensorflow import keras

from cnn_res_degrader.utils import img_as_batch, make_comparison_fig
from cnn_res_degrader.data_loading import load_proba_img_as_array


class InferenceImagePreview(keras.callbacks.Callback):
    def __init__(self, hr_path: Path, lr_path: Path, output_dir: Path):
        self._hr_img = load_proba_img_as_array(hr_path)
        self._lr_img = load_proba_img_as_array(lr_path)
        self._batch_input = img_as_batch(self._hr_img)
        self._output_dir = output_dir

    def on_epoch_end(self, epoch, logs=None):
        img_pred = self.model(self._batch_input)[0]
        cmpr_fig = make_comparison_fig(self._hr_img, self._lr_img, img_pred)
        output_path = self._output_dir/f'{self.model.name}-epoch_{epoch+1}.png'
        print(f'Saving inference preview at: {output_path}.')
        cmpr_fig.savefig(output_path, dpi=300)
