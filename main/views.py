from django.shortcuts import render
from django.http import  HttpResponse
from . import tfidf
# Create your views here.
from .models import Document

def index(request):
    return render(request,'index.html')

def doc_by_id(request,doc):

    doc1 = Document.objects.get(name=doc)
    if(doc1):
        return render(request,'doc_details.html',{'doc':doc1})
    else:
        return render(request,'doc_details.html',{})

def search_doc(request):
    if(request.method=="POST"):
        query = request.POST.get("query", "")
        li = tfidf.start(query)
        # res=''
        # for doc in li:
        #     res+= doc
        return render(request,'search_doc.html',{'query':query,'docList':li})
    else:
        return render(request,'search_doc.html',{})  
