from rest_framework import serializers

from ai_rest.models import PredictImage


class PrecitSerializer(serializers.ModelSerializer):
    class Meta:
        model = PredictImage
        fields = ("image",)
