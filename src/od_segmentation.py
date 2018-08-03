import numpy as np
import os
import tensorflow as tf
from PIL import Image,ImageDraw
import cv2
import math
import logging

logging.basicConfig(format='%(levelname)s:%(asctime)s:%(filename)s:%(lineno)d:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def initNet(networkDir):
    """
    This method is used to initialize a network.
    Args:
        networkDir: Directory containg network details like frozen graph, label mapping etc

    Returns: A dictionary containing details of network for fast lookup

    """
    logger.info('Initializing tensorflow model from: {}'.format(networkDir))
    net_details_dict = {}

    # Read required files from network directory
    graph_filepath = os.path.join(networkDir, 'frozen_inference_graph.pb')
    if not os.path.exists(graph_filepath):
        raise IOError('frozen graph path: {} does not exist'.format(graph_filepath))
    labels_file_path = os.path.join(networkDir, 'label.txt')
    if not os.path.exists(graph_filepath):
        raise IOError('labels mapping file path: {} does not exist'.format(labels_file_path))
    labels_dict = {}

    if net_details_dict.has_key('loadedGraph'):
        return net_details_dict['loadedGraph'],net_details_dict['graph'],net_details_dict['labelDict']
    else:
        with tf.gfile.GFile(graph_filepath, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with open(labels_file_path, 'r') as f:
            for kv in [d.strip().split(':') for d in f]:
                labels_dict[int(kv[0])] = kv[1]

        graph = tf.get_default_graph()

        tf.import_graph_def(graph_def)
        # for op in graph.get_operations():
        #     print op.name

        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        sess = tf.Session()
        # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # Create variables for output tensors
        image_tensor = graph.get_tensor_by_name('import/image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = graph.get_tensor_by_name('import/detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = graph.get_tensor_by_name('import/detection_scores:0')
        classes = graph.get_tensor_by_name('import/detection_classes:0')
        num_detections = graph.get_tensor_by_name('import/num_detections:0')

        # Create dictionary of all the details
        net_details_dict['session'] = sess
        # net_details_dict['graph'] = graph
        net_details_dict['labelDict'] = labels_dict
        net_details_dict['image_tensor'] = image_tensor
        net_details_dict['boxes'] = boxes
        net_details_dict['scores'] = scores
        net_details_dict['classes'] = classes
        net_details_dict['num_detections'] = num_detections
        logger.info('Initialized tensorflow model from: {}'.format(networkDir))
        return net_details_dict

def load_image_into_numpy_array(imagePath):
    try:
        image = cv2.imread(imagePath)
        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image_np
    except:
        logger.exception('Exception occurred while opening the image: {} using opencv, trying with PIL now!'.format(imagePath))
        try:
            image = Image.open(imagePath)
            (im_width, im_height) = image.size
            return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
        except:
            logger.exception('Exception occurred while opening the image: {} using PIL!'.format(imagePath))
            raise

# Rendering stuff initialization
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (0, 255, 0)
lineType = 2
# def renderImageSegments(imagePath, renderDir, boxOfImage, classesInImage, scoreOfImage):
def renderImageSegments(imageLoaded, imageName, renderDir, boxOfImage, classesInImage, scoreOfImage):
    if not os.path.exists(renderDir):
        logger.info('Creating directory: {} to store rendered segments'.format(renderDir))
        os.makedirs(renderDir)
    # imageName = imagePath[imagePath.rfind('/')+1:]
    # imageLoaded = cv2.imread(imagePath)
    imageLoaded = cv2.cvtColor(imageLoaded, cv2.COLOR_RGB2BGR)
    height, width, channels = imageLoaded.shape
    renderedImagePath = os.path.join(renderDir, imageName)
    for idxI, box in enumerate(boxOfImage):
        classs = classesInImage[idxI]
        score = scoreOfImage[idxI]
        ymin = int(math.floor(box[0] * height))
        xmin = int(math.floor(box[1] * width))
        ymax = int(math.ceil(box[2] * height))
        xmax = int(math.ceil(box[3] * width))
        cv2.rectangle(imageLoaded, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
        # cv2.putText(imageLoaded, str(classs), (xmax, ymax), font, fontScale, fontColor, lineType)
        cv2.putText(imageLoaded, '{}_{}'.format(classs, score), (xmax, ymax), font, fontScale, fontColor, lineType)
    cv2.imwrite(renderedImagePath, imageLoaded)

def runObjectDetectionOnImages(images, netDetailsDict, threshold=0.6):
    # Read pre-filled details of variables from the provided dictionary
    session = netDetailsDict['session']
    # graphInfo = netDetailsDict['graph']
    labelsMap = netDetailsDict['labelDict']
    image_tensor = netDetailsDict['image_tensor']
    boxes = netDetailsDict['boxes']
    scores = netDetailsDict['scores']
    classes = netDetailsDict['classes']
    num_detections = netDetailsDict['num_detections']

    imageMapHash = {}

    # Run actual detection on the batch.
    # (boxes, scores, classes, num_detections) = session.run([boxes, scores, classes, num_detections], feed_dict={image_tensor: images})
    (boxes, scores, classes, num_detections) = session.run([boxes, scores, classes, num_detections], feed_dict={image_tensor: images})

    # Filter based on score threshold #TODO: check if we get score for all the images even if there are no detections
    for idx,scoresOfimageToConsider in enumerate(scores):
        imageMapHash[idx] = []
        boxesOfImage = boxes[idx]
        classesOfImage = classes[idx]
        # classesOfImage = [labelsMap[c] for c in classesOfImage] # Note: uncomment this if we want string labels
        for idxInner,score in enumerate(scoresOfimageToConsider):
            if score > threshold:
                imageMapHash[idx].append([boxesOfImage[idxInner],classesOfImage[idxInner],score])

        imageMapHash[idx] = np.asarray(imageMapHash[idx])

    # valid_indices = np.where(scores > threshold)
    # scores = scores[valid_indices]
    # boxes = boxes[valid_indices]
    # classes = classes[valid_indices]
    return imageMapHash
    # return boxes,classes,scores,valid_indices#,np.asarray(images)[valid_indices]

def runObjectDetectionOnFolder(imageFolder, netDetailsDict, renderDir=None, batch_size=8):
    """

    Args:
        imageFolder:
        netDetailsDict:
        renderDir:

    Returns:

    """
    # Create a map of image name and its actual data, so that we don't have to load it multiple times
    # imageNameDataMap = {}
    # for imageName in os.listdir(imageFolder):
    #     imagePath = os.path.join(imageFolder, imageName)
    #     imageNameDataMap[imageName] = load_image_into_numpy_array(imagePath)
    # batch_size = len(imageNameDataMap.keys())  # 16
    # imagesList = imageNameDataMap.values()

    logger.info('Running OD on: {}'.format(imageFolder))
    imageNames = os.listdir(imageFolder)
    #len(imageNames)
    # Iterate over batches
    for imageNamesBatch in [imageNames[x : x + batchSize] for x in xrange(0, len(imageNames), batchSize)]:
        logger.info('\tRunning OD on batch of {} images'.format(len(imageNamesBatch)))
        # Create a map of image name and its actual data, so that we don't have to load it multiple times
        imageNameDataMap = {}
        for imageName in imageNamesBatch:
            try:
                imagePath = os.path.join(imageFolder, imageName)
                imageNameDataMap[imageName] = load_image_into_numpy_array(imagePath)
            except:
                pass

        # Run OD over this batch
        imageDataBatch = imageNameDataMap.values()
        imageMapHash = runObjectDetectionOnImages(imageDataBatch, netDetailsDict)
        logger.info('\tRan OD on batch of {} images, going to write their segments on filesystem'.format(len(imageDataBatch)))

        # Iterate over every image of this batch and write its segments to a text file
        # Note: we are using imageNameDataMap.keys() instead of imageNamesBatch because
        # when batch is created using imageNameDataMap.values(), order of elements might
        # be different from that in imageNamesBatch
        for idx,imageName in enumerate(imageNameDataMap.keys()):
            infoAboutImage = imageMapHash[idx]
            if len(infoAboutImage) == 0:
                continue
            boxOfImage = imageMapHash[idx][:,0]
            classesInImage = imageMapHash[idx][:, 1]
            scoreOfImage = imageMapHash[idx][:,2]
            # boxOfImage = boxes[idx]
            # imageToLoad = imagesPathList[valid_indices[0][idx]]
            # scoreOfImage = scores[idx]

            baseName = imageName[:imageName.rfind('.')]
            predFileName = baseName + '.txt'
            predFilePath = os.path.join(imageFolder, predFileName)
            height, width, channels = imageNameDataMap[imageName].shape
            with open(predFilePath, 'w') as predFile:
                for idxI, box in enumerate(boxOfImage):
                    classs = classesInImage[idxI]
                    score = scoreOfImage[idxI]
                    ymin = int(math.floor(box[0] * height))
                    xmin = int(math.floor(box[1] * width))
                    ymax = int(math.ceil(box[2] * height))
                    xmax = int(math.ceil(box[3] * width))
                    line = '{},{},{},{},{},{}\n'.format(xmin, ymin, xmax, ymax, score, classs)
                    predFile.write(line)

            # Render segments of this image
            if renderDir:
                renderImageSegments(imageNameDataMap[imageName], imageName, renderDir, boxOfImage, classesInImage, scoreOfImage)

        logger.info('\tWrote segments of batch of {} images to filesystem'.format(len(imageDataBatch)))
    logger.info('Ran OD on: {}'.format(imageFolder))


if __name__ == '__main__':
    networkDir = '/home/neo/ML/tensorflow_models/models/research/object_detection/models/frozen_graphs/dialysis_combined_18072018'
    imageDir = '/media/neo/krypton/data/data_dcdc/batch2'
    renderDir = '/media/neo/krypton/data/data_dcdc/batch2_renderedImages'
    netDetailsDict = initNet(networkDir)
    runObjectDetectionOnFolder(imageDir, netDetailsDict, renderDir=renderDir,batch_size=4)
    pass