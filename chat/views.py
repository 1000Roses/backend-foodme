from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework.decorators import action, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from ml.transformer.main import inference as transformer_inference
from ml.decisionTree.main import inference as decision_tree_inference


class ChatViewSet(viewsets.ViewSet):
    parser_classes = [MultiPartParser, FormParser, JSONParser]

    @action(detail=False, methods=['GET'])
    def chat_normal(self, request):
        question = request.GET.get("question", None)
        print("question: ", question)
        answer = transformer_inference(question)
        answer = answer.replace("[start]", "").replace("[end]", "").replace("_", " ").strip()
        print("answer normal chat: ", answer)
        return Response(answer)

    
    @action(detail=False, methods=['GET'])
    def suggest_food(self, request):
        tam_trang   = request.GET.get("tam_trang", "bình thường")
        tinh_trang = request.GET.get("tinh_trang", "bình thường")
        the_trang = request.GET.get("the_trang", "bình thường")
        khau_vi = request.GET.get("khau_vi", "ngọt")
        thoi_diem  = request.GET.get("thoi_diem", "sáng")
        food_suggest = decision_tree_inference([tam_trang, tinh_trang, the_trang, khau_vi, thoi_diem])
        return Response(food_suggest)