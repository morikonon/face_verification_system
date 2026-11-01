from deepface import DeepFace

# Function to get face embedding
def get_embedding(image_path: str, model_name: str = "Facenet512"):
	result = DeepFace.represent(img_path=image_path, model_name=model_name)

	embedding = result[0]["embedding"]
	facial_area = result[0]["facial_area"]
	face_confident = result[0]["face_confidence"]

	return {
		"embedding": embedding, 
		"facial_area": facial_area, 
		"face_confidence": face_confidence
	}

