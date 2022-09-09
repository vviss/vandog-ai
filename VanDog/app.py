import os
import gc
import logging
import tracemalloc
from keras import backend as K

import random
from flask import Flask, request, render_template
# from werkzeug.utils import secure_filename

from stylizer import VanDog, InferenceConfig, model_dir
import Mask_RCNN.mrcnn.model as modellib

app = Flask(__name__)

# Garbage collection
# collected = gc.collect()
# logging.warn("From app.py - Garbage collector: collected",
#     "%d objects." % collected)

# create instance
ROOT_DIR =  os.getcwd()
print('W-- from app.py - current working directory', ROOT_DIR)

app.config['UPLOAD_FOLDER'] = ROOT_DIR+'/static'  

segmenter_path = os.path.join(ROOT_DIR, "Mask_RCNN/mask_rcnn_coco.h5")
cartoonizer_path = os.path.join(ROOT_DIR, "Cartoonizer/test_code/saved_models")

random_id = ''.join([random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789') for i in range(10)])
logging.info('Random ID generated:', random_id)
pet_filename = 'PET_' + random_id + '.jpg'
bg_filename = 'BG_' + random_id + '.jpg'
output_filename = 'VanDog-Stylizer_' + random_id + '.jpg'

# pet_filepath = os.path.join(app.config['UPLOAD_FOLDER'], pet_filename)
pet_filepath = os.path.join('static', pet_filename)
bg_filepath = os.path.join('static', bg_filename)
output_filepath = os.path.join('static', output_filename)
zaka_logo_filepath = os.path.join('static', 'zaka_logo.png')


logging.basicConfig(level=logging.INFO)

@app.route("/")
def index():
    return render_template('index.html', show_input=0, show_results=0, zaka_logo_filepath=zaka_logo_filepath)


@app.route("/upload", methods=["POST", 'GET'])
def upload():
    # K.clear_session()
    logging.info("Stylize request received!")

    if request.method == 'POST':
        f_pet = request.files['pet_image']
        # pet_filename = secure_filename(f_pet.filename)
        f_pet.save(pet_filepath)
        print('Loaded pet image')

        f_bg = request.files['bg_image']
        # bg_filename = secure_filename(f_bg.filename)
        f_bg.save(bg_filepath)
        print('Loaded background image')
    
    return render_template('index.html', pet_filepath=pet_filepath, bg_filepath=bg_filepath, show_input=1, show_results=0)

@app.route("/stylize", methods=["POST", 'GET'])
def stylize():
    tracemalloc.start()
    # K.clear_session()
    snapshot_1 = tracemalloc.take_snapshot()

    logging.info("Stylize request received!")
    
    logging.info('Initizalizing stylizer instance')
    # stylizer = VanDog(segmenter_path=segmenter_path, cartoonizer_path=cartoonizer_path)
    # stylizer = VanDog(segmenter=segmenter, cartoonizer_path=cartoonizer_path)
    stylizer.get_stylized(pet_filepath, bg_filepath, output_filepath)
    
    # del stylizer

    # print('W--static dir', [os.path.join(dp, f) for dp, dn, fn in os.walk(os.getcwd()) for f in fn])

    # Memory allocation monitoring
    snapshot_2 = tracemalloc.take_snapshot()
    top_stats = snapshot_2.compare_to(snapshot_1, 'lineno')
    print("[ Top 10 differences in memory after app run]")
    for stat in top_stats[:10]:
        print(stat)

    return render_template('index.html', output_filepath=output_filepath, pet_filepath=pet_filepath, bg_filepath=bg_filepath, show_input=1, show_results=1)

segmenter = None
def load_segmenter():
    global segmenter

    config = InferenceConfig()
    segmenter = modellib.MaskRCNN(mode="inference", model_dir=model_dir, config=config)
    segmenter.load_weights(segmenter_path, by_name=True)
    segmenter.keras_model._make_predict_function()

stylizer = None
def load_stylizer():
    global stylizer
    stylizer = VanDog(segmenter=segmenter, cartoonizer_path=cartoonizer_path)


def main():
    """Run the Flask app."""
    load_segmenter()
    load_stylizer()

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port) 


if __name__ == "__main__":
    main()
