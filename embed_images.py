import os
import uuid
from embedder import get_embedding
from vector_database import create_db, add_vector, search_vector
from qdrant_client.models import PointStruct
from config import Config

current_path = os.getcwd()
images_list = os.listdir("images")

create_db(collection_name=Config.collection_name, vector_size=Config.vector_size)

for image in images_list:
	# image path
	image_path = os.path.join(current_path, image)
	
	# get embedding
	embedding, facial_area, face_confidence = get_embedding(image_path, model_name=Config.model_name)

	if face_confidence < 0.6:
		continue

	point = PointStruct(
		id=uuid.uuid4(),
		vector=embedding,
		payload=facial_area
	)

	add_vector(point, collection_name=Config.collection_name)


