from pathlib import Path
import numpy as np
import tensorflow as tf


def convert_to_tflite(model, output_path: Path):
    print(f"Converting model to TensorFlow Lite at {output_path}...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Add this line to include TensorFlow operations
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]

    # Disable experimental lowering of tensor list operations
    converter._experimental_lower_tensor_list_ops = False

    tflite_model = converter.convert()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Model converted to TensorFlow Lite and saved at {output_path}")


class TFLiteModelWrapper:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, input_data, verbose=False):
        # Ensure input data has the correct shape and type
        for i, detail in enumerate(self.input_details):
            input_shape = detail['shape']
            input_data[i] = np.array(input_data[i], dtype=np.float32)  # Ensure the data type is FLOAT32
            if input_data[i].shape != tuple(input_shape):
                input_data[i] = np.reshape(input_data[i], input_shape)
            self.interpreter.set_tensor(detail['index'], input_data[i])

        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data
