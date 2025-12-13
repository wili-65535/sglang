import os
import requests
import time


def call_api(prompt, name="test", **kwargs):
    """调用sglang视频生成API（异步模式）"""
    host = os.environ.get("CACHE_DIT_HOST", "localhost")
    port = int(os.environ.get("CACHE_DIT_PORT", 8000))
    base_url = f"http://{host}:{port}"
    
    # Wan2.2-T2V-A14B 默认参数：720P, 81帧, 5秒视频
    num_frames = kwargs.get("num_frames", 81)
    fps = kwargs.get("fps", 16)
    seconds = int(num_frames / fps)
    
    # 构建size参数 (格式: "heightxwidth")
    width = kwargs.get("width", 1280)
    height = kwargs.get("height", 720)
    size = f"{height}x{width}"
    
    payload = {
        "prompt": prompt,
        "size": size,
        "seconds": seconds,
        "fps": fps,
        "num_frames": num_frames,
        "seed": kwargs.get("seed", 1234),
        "extra_body": {
            "num_inference_steps": kwargs.get("num_inference_steps", 27),
            "guidance_scale": kwargs.get("guidance_scale", 3.5),
            "guidance_scale_2": kwargs.get("guidance_scale_2", 4.0),
        }
    }

    if "negative_prompt" in kwargs:
        payload["extra_body"]["negative_prompt"] = kwargs["negative_prompt"]

    try:
        # 步骤1: 创建视频生成任务
        print(f"创建任务: {name}...")
        response = requests.post(f"{base_url}/v1/videos", json=payload, timeout=30)
        response.raise_for_status()
        job = response.json()
        video_id = job["id"]
        print(f"任务ID: {video_id}, 状态: {job['status']}")
        
        # 步骤2: 轮询任务状态
        max_wait_time = 600  # 最多等待10分钟
        start_time = time.time()
        poll_interval = 5  # 每5秒查询一次
        
        while time.time() - start_time < max_wait_time:
            response = requests.get(f"{base_url}/v1/videos/{video_id}", timeout=10)
            response.raise_for_status()
            job = response.json()
            status = job["status"]
            progress = job.get("progress", 0)
            
            print(f"状态: {status}, 进度: {progress}%")
            
            if status == "completed":
                print("生成完成！")
                break
            elif status == "failed":
                error_msg = job.get("error", {}).get("message", "Unknown error")
                print(f"生成失败: {error_msg}")
                return None
            
            time.sleep(poll_interval)
        else:
            print(f"超时: 等待超过{max_wait_time}秒")
            return None
        
        # 步骤3: 下载视频文件
        print("下载视频...")
        response = requests.get(f"{base_url}/v1/videos/{video_id}/content", timeout=30)
        response.raise_for_status()
        
        filename = f"{name}.mp4"
        with open(filename, "wb") as f:
            f.write(response.content)
        
        print(f"保存成功: {filename}")
        return filename

    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
        return None
    except Exception as e:
        print(f"未知错误: {e}")
        return None


def test_basic():
    """基础测试：720P, 81帧, 5秒视频"""
    return call_api(
        prompt="A cat walks on the grass, realistic", 
        name="wan_t2v_basic"
    )


def test_custom_prompt():
    """自定义prompt测试"""
    return call_api(
        prompt="A beautiful sunset over the ocean with waves crashing on the shore",
        name="wan_t2v_sunset",
        seed=42,
    )


def test_with_negative_prompt():
    """带negative prompt的测试"""
    # Wan2.2默认的中文negative prompt
    negative_prompt = (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
        "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
        "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
        "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    )

    return call_api(
        prompt="A dog running in a park, high quality, realistic",
        negative_prompt=negative_prompt,
        name="wan_t2v_negative",
        seed=999,
    )


def test_short_video():
    """短视频测试：2秒视频（33帧）"""
    return call_api(
        prompt="A bird flying in the sky",
        name="wan_t2v_short",
        num_frames=33,  # 2秒视频 (33帧 / 16fps ≈ 2秒)
        num_inference_steps=20,  # 减少步数以加快生成
        seed=777,
    )


def test_different_resolution():
    """不同分辨率测试：保持720P但调整宽高比"""
    return call_api(
        prompt="A car driving on a highway",
        name="wan_t2v_resolution",
        width=1280,
        height=720,
        seed=555,
    )


def test_dual_guidance():
    """测试Wan2.2的双guidance scale特性"""
    return call_api(
        prompt="A majestic eagle soaring through the clouds",
        name="wan_t2v_dual_guidance",
        guidance_scale=3.5,  # high_noise guidance
        guidance_scale_2=4.0,  # low_noise guidance
        seed=888,
    )


if __name__ == "__main__":
    print("开始测试 Wan2.2-T2V-A14B 模型...")
    print("配置: 720P (1280x720), 81帧, 16fps (5秒视频), 27步推理\n")
    
    test_basic()
    test_custom_prompt()
    test_with_negative_prompt()
    test_short_video()
    test_different_resolution()
    test_dual_guidance()
    
    print("\n所有测试完成!")
