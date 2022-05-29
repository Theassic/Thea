from django.shortcuts import render

from django.conf import settings
from django.http import HttpResponse
import os
import time



def index(request):
    return render(request, "index.html")


def files(request):
    # 由前端指定的name获取到图片数据
    video = request.FILES.get('video')
    # 获取图片的全文件名
    img_name = video.name
    # 截取文件后缀和文件名
    mobile = os.path.splitext(img_name)[0]
    ext = os.path.splitext(img_name)[1]
    ts = int(time.time())
    # 重定义文件名
    # img_name = f'avatar-{mobile}{ext}'
    vid_name = f'{ts}{ext}'
    # 从配置文件中载入图片保存路径
    img_path = os.path.join(settings.IMG_UPLOAD, img_name)
    # 写入文件
    with open(img_path, 'ab') as fp:
        # 如果上传的图片非常大，就通过chunks()方法分割成多个片段来上传
        for chunk in video.chunks():
            fp.write(chunk)
    return HttpResponse('uploads success')

def uploadVid(request):
    if request.method == 'POST':
        # 图片资源所属文章的id
        article_id = request.POST.get('article_id')
        # 提交过来的类型为formdata
        file_obj = request.FILES.get('videoinput')
        # size = file_obj.size
        # if size > 30 * 1024 * 1024:  # 限制输入大小为30M
        #     return HttpResponse(json.dumps({'code': 405, 'information': '上传视频或图片大于30M！'}),
        #                         content_type="application/json")
        ts = int(time.time())

        # video_name = ts + '.' + file_obj.name.split('.')[-1]
        #     try:
        #         with open(video_name, 'wb+') as f:
        #             f.write(file_obj.read())
        #     except Exception:
        #         result = {"code": '500', 'information': '文件写入错误！'}
    #             return HttpResponse(json.dumps(result, ensure_ascii=False))
    #
    # return HttpResponse(json.dumps(result, ensure_ascii=False), content_type="application/json")
