
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .llm_manager import LLMManager
import asyncio

@csrf_exempt
async def chat(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        message = data.get('message')
        selected_llms = data.get('selected_llms', [])
        
        llm_manager = LLMManager()
        responses = {}
        
        # Process responses concurrently
        tasks = [
            llm_manager.get_response(llm_type, message)
            for llm_type in selected_llms
        ]
        
        results = await asyncio.gather(*tasks)
        
        for llm_type, response in zip(selected_llms, results):
            responses[llm_type] = response
        
        return JsonResponse({'responses': responses})

    return JsonResponse({'error': 'Invalid request method'})

def index(request):
    return render(request, 'chat/index.html')