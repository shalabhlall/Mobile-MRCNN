import os
import sys
from skimage.io import imread

# Root directory of the project
ROOT_DIR = os.getcwd()
Data_DIR = "images/"

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.config import Config

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class_names = ['BG', 'car']


class ShapesConfig(Config):
    """Configuration for training on the dataset.
    Derives from the base Config class and overrides values specific
    to the dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 1 images per GPU. We can put multiple images on each
    # GPU. Batch size is (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + nucleus

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 64, 128, 256)  # anchor side in pixels

    # Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 800

    # set number of epoch
    STEPS_PER_EPOCH = 200
    

model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes3.h5")

class InferenceConfig1(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.9


inference_config = InferenceConfig1()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)


# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
model.load_weights(model_path, by_name=True)

# pic.dump(model, open('model1.pkl', 'wb'))

def prediction():
    test_ids = next(os.walk(IMAGE_DIR))
    print(test_ids[2])
    for image_id in test_ids[2]:
        image = os.path.join(IMAGE_DIR, image_id)
        img = imread(image)
        
        results = model.detect([img], verbose=1)
    
        r = results[0]
        
        image_name=image.split(".")[0]
        if image_name[-3:] != '_IS':
            imag = visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
            imag.savefig(image_name+'_IS.jpg', bbox_inches = 'tight', pad_inches = 0)


from flask import Flask, render_template, request, send_from_directory


app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def home():
    return render_template('upload.html')

# Route to upload image
@app.route("/upload", methods=["POST", 'GET'])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    
    for upload in request.files.getlist("file"):
        print(upload)
        filename1 = upload.filename
        destination = "/".join([target,filename1])
        print(destination)
        upload.save(destination)
    prediction()
    image_names = os.listdir('./images')
    return render_template("index.html", image_names=image_names)

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)


if __name__ == "__main__":
    app.run(debug=True)

