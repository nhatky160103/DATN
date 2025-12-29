from collections import Counter
from .infer_image import getEmbedding
from .identity_person import find_closest_person
from .utils import load_onnx_model


arcface_model = load_onnx_model()

def check_validation(
        input, 
        embeddings, 
        image2class, 
        index2class,
        config=None,
        ):
    is_anti_spoof = config['infer_video']['is_anti_spoof']
    validation_threshold = config['infer_video']['validation_threshold']
    anti_spoof_threshold = config['infer_video']['anti_spoof_threshold'] 
    distance_mode = config['identity_person']['distance_mode']
    l2_threshold = config['identity_person']['l2_threshold']
    cosine_threshold = config['identity_person']['cosine_threshold']

    valid_images = input['valid_images']
    is_reals = input.get('is_reals', [])
    guide_base = '/static/audio/'

    if valid_images is None or len(valid_images) == 0:
        print("Không có ảnh để xử lý.")
        return 'UNKNOWN', guide_base + 'retry.mp3'

    if embeddings is None or len(embeddings) == 0 or not image2class or not index2class:
        print("Không có nhân viên trong database.")
        return 'UNKNOWN', guide_base + 'retry.mp3'
    
    predict_class = []
    processed_faces = []

    print("START TESTING ......................................")
    for i, face in enumerate(valid_images):
        if is_anti_spoof and is_reals:
            if not is_reals[i][0] and is_reals[i][1] > anti_spoof_threshold:
                continue
        processed_faces.append(face)

    # Get embeddings for all processed faces in batch
    if processed_faces:
        pred_embeds = getEmbedding(arcface_model, processed_faces)
        for pred_embed in pred_embeds:
            result = find_closest_person(
                pred_embed, 
                embeddings, 
                image2class, 
                distance_mode=distance_mode, 
                l2_threshold=l2_threshold, 
                cosine_threshold=cosine_threshold
            )
            if result != -1:
                predict_class.append(result)
            print(result)
            print("__"*10)

    # Count predictions and check against validation threshold
    class_count = Counter(predict_class)
    majority_threshold = len(valid_images) * validation_threshold

    for cls, count in class_count.items():
        if count >= majority_threshold:
            person_id = index2class.get(cls, 'UNKNOWN')
            print(f"✅ Nhận diện thành công: {person_id}")
            return person_id, guide_base + 'greeting.mp3'

    print("❌ Unknown person")
    return 'UNKNOWN', guide_base + 'retry.mp3'





