import gc
import logging
import cv2
import numpy as np
import tensorflow as tf 
from keras import backend as K
from Cartoonizer.test_code.network import unet_generator
from Cartoonizer.test_code.guided_filter import guided_filter


def resize_crop(image):
    h, w, c = np.shape(image)
    print('Wis-FROM RESIZE_CROP - before:', image.shape)

    if min(h, w) > 720:
        if h > w:
            h, w = int(720*h/w), 720
        else:
            h, w = 720, int(720*w/h)


    image = cv2.resize(image.astype('float32'), (w, h),
                       interpolation=cv2.INTER_AREA)
    h, w = (h//8)*8, (w//8)*8
    image = image[:h, :w, :]

    print('Wis-FROM RESIZE_CROP - after:', image.shape)

    return image
    

def cartoonize(image, model_path):
    print('FROM CARTOONIZE - model path >>>', model_path)
    # K.clear_session()

    # load_folder = 'test_images' ##replaced arg by image
    # save_folder = 'cartoonized_images'
    # if not os.path.exists(save_folder):
    #     os.mkdir(save_folder)

    input_photo = tf.placeholder(tf.float32, [1, None, None, 3])
    network_out = unet_generator(input_photo)
    final_out = guided_filter(input_photo, network_out, r=1, eps=5e-3)

    all_vars = tf.trainable_variables()
    gene_vars = [var for var in all_vars if 'generator' in var.name]
    saver = tf.train.Saver(var_list=gene_vars)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint(model_path))

    output = None

    try:
        # image = cv2.imread(image)
        image = resize_crop(image)
        batch_image = image.astype(np.float32)/127.5 - 1
        batch_image = np.expand_dims(batch_image, axis=0)

        output = sess.run(final_out, feed_dict={input_photo: batch_image})
        output = (np.squeeze(output)+1)*127.5
        output = np.clip(output, 0, 255).astype(np.uint8)
    except Exception as e:
        print('\n\nERROR FROM CARTOONIZE:' , e)

    # K.clear_session()
    print('Wis-SESSION CLEARED')

    # Garbage collection to deal with memory issues
    # del network_out
    # del input_photo
    # del final_out    
    # collected = gc.collect()
    # logging.warn("From cartoonize.py - Garbage collector: collected",
    #       "%d objects." % collected)

    return output


    

if __name__ == '__main__':
    pass
    

    