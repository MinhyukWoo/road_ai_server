import io
from django.shortcuts import render
from django import http

# Create your views here.
from rest_framework import generics
from ai_rest.models import PredictImage
from ai_rest.serializers import PrecitSerializer
import tensorflow as tf
from rest_framework.response import Response

import numpy as np

from rest_framework import status
import os

import json


def get_class_to_sign():
    with open("class_to_sign.json", "r") as file:
        json_data = json.load(file)
    out = {item["model_class_category"]: item["sign_category"] for item in json_data}
    return out


model = tf.keras.models.load_model("final.h5")
class_to_sign = get_class_to_sign()


class PredictView(generics.CreateAPIView):
    serializer_class = PrecitSerializer
    name = "predict"

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)

        image_path = os.path.join("media", serializer.data["image"].split("/")[-1])

        test_image = tf.keras.utils.load_img(image_path, target_size=(150, 150))
        test_image = tf.keras.utils.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image /= 255.0

        predictions = model.predict(test_image)
        prediction_num = np.argmax(predictions)
        road_num = class_to_sign[prediction_num]
        out_dict = {"request_result": road_num}
        out_json_str = json.dumps(out_dict)
        PredictImage.objects.all().delete()
        os.remove(image_path)
        return Response(
            json.loads(out_json_str), status=status.HTTP_201_CREATED, headers=headers
        )
