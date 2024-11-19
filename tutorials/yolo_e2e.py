import re
import os
import numpy
import argparse
from pathlib import Path
import onnxruntime_extensions


def get_yolo_model(version: int, onnx_model_name: str):
    # install yolo11l
    from pip._internal import main as pipmain
    try:
        import ultralytics
    except ImportError:
        pipmain(['install', 'ultralytics'])
        import ultralytics
    pt_model = Path(onnx_model_name).with_suffix(".pt")
    model = ultralytics.YOLO(str(pt_model))  # load a pretrained model
    exported_filename = model.export(format="onnx")  # export the model to ONNX format
    assert exported_filename, f"Failed to export {pt_model} to onnx"
    import shutil
    shutil.move(exported_filename, onnx_model_name)

def add_pre_post_processing_to_yolo(input_model_file: Path, output_model_file: Path, output_image: bool):
    """Construct the pipeline for an end2end model with pre and post processing. 
    The final model can take raw image binary as inputs and output the result in raw image file.

    Args:
        input_model_file (Path): The onnx yolo model.
        output_model_file (Path): where to save the final onnx model.
        output_image (bool): If post processing should ouput image or bounding box locations.
    """
    from onnxruntime_extensions.tools import add_pre_post_processing_to_model as add_ppp
    add_ppp.yolo_detection(input_model_file, output_model_file, "jpg", onnx_opset=19, output_as_image=output_image)

def run_inference(onnx_model_file: Path, output_image: bool):
    import onnxruntime as ort
    import numpy as np

    providers = ['CPUExecutionProvider']
    session_options = ort.SessionOptions()
    session_options.register_custom_ops_library(onnxruntime_extensions.get_library_path())

    image = np.frombuffer(open(f'{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/test/data/ppp_vision/wolves.jpg', 'rb').read(), dtype=np.uint8)
    session = ort.InferenceSession(str(onnx_model_file), providers=providers, sess_options=session_options)

    # Print all output names to determine what the model provides
    output_names = [output.name for output in session.get_outputs()]
    print(f"Model output names: {output_names}")

    inname = [i.name for i in session.get_inputs()]
    inp = {inname[0]: image}

    # Run inference using the dynamically determined output names
    output = session.run(output_names, inp)

    if output_image:
        # Assuming you want to work with the first output
        output_filename = './result.jpg'
        open(output_filename, 'wb').write(output[0])  # Save the first output

        from PIL import Image
        Image.open(output_filename).show()
    else:
        print("Not outputting an image...")
        print(output)

if __name__ == '__main__':
    class Formatter(argparse.ArgumentDefaultsHelpFormatter, #prints default values
                    argparse.RawDescriptionHelpFormatter # add new lines in description
                    ): pass
    parser = argparse.ArgumentParser(description="Convert Yolo model to onnx format including pre and post processing.\n"
    "Will fisrt download the '.pt' file if it doesn't exist.\n"
    "The download and output model will be in the current working directory", 
                                     formatter_class=Formatter)
    parser.add_argument('--model', type=str, default='yolo11l', help='model to download and export')
    parser.add_argument('--output-dir', type=str, help='output directory, if not supplied will use current working directory')
    parser.add_argument('--create-e2e-model', action=argparse.BooleanOptionalAction, default=True, help='create the end to end onnx model')
    parser.add_argument('--image', action=argparse.BooleanOptionalAction, default=False, help='output an image with bounding boxes, instead of box locations')
    parser.add_argument('--run-inference', action=argparse.BooleanOptionalAction, default=False, help='test inference')
    args = parser.parse_args()

    # YOLO version. Tested with 5, 8, and 11.
    onnx_model_name = Path(f"./{args.model}.onnx")
    matches = re.findall(r"\d+", args.model)
    assert(len(matches) == 1), "Model should have only one version number in it"
    version = int(matches[0])
    onnx_e2e_model_name = Path(f"{args.model}_with_pre_post_processing.onnx")

    if args.output_dir:
        os.chdir(args.output_dir)

    if args.create_e2e_model:
        if not onnx_model_name.exists():
            print("Fetching original model...")
            get_yolo_model(version, str(onnx_model_name))
        print("Adding pre/post processing...")
        add_pre_post_processing_to_yolo(onnx_model_name, onnx_e2e_model_name, args.image)

    if args.run_inference:
        print("Testing updated model...")
        run_inference(onnx_e2e_model_name, args.image)
