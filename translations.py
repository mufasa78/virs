# Translations for both Flask and Streamlit applications

translations = {
    'zh': {
        # Page titles and headers
        'page_title': '深度学习视频图像修复系统',
        'page_subtitle': '使用我们的深度学习模型提升视频质量',
        
        # Model settings
        'model_settings': '模型设置',
        'model_checkpoint': '模型检查点路径',
        'model_loaded': '模型加载成功！',
        'using_untrained': '使用未训练的模型 - 未找到检查点',
        'using_device': '使用设备：',
        
        # Processing parameters
        'processing_params': '处理参数',
        'sequence_length': '序列长度',
        'sequence_length_help': '一次处理的帧数',
        'frame_overlap': '帧重叠',
        'frame_overlap_help': '序列之间的重叠帧数',
        
        # Upload section
        'upload_video': '上传视频',
        'choose_video': '选择视频文件',
        'supported_formats': '支持的格式：MP4、AVI、MOV、MKV、WEBM（最大大小：500MB）',
        
        # Processing status
        'processing': '处理中',
        'extracting_frames': '正在提取视频帧...',
        'extracted_frames': '已提取 {} 帧',
        'processing_frames': '正在处理帧，序列长度 {}，重叠 {}...',
        'creating_video': '正在创建输出视频...',
        'processing_complete': '视频处理完成！',
        
        # Results section
        'results': '结果',
        'original_video': '原始视频',
        'enhanced_video': '增强视频',
        'download_enhanced': '下载增强视频',
        'process_another': '处理另一个视频',
        
        # Error messages
        'error': '错误',
        'try_again': '重试',
        'load_error': '加载模型时出错：{}',
        'process_error': '处理视频时出错：{}',
        
        # Footer
        'footer': '深度学习视频图像修复系统 © 2023'
    },
    'en': {
        # Page titles and headers
        'page_title': 'Deep Learning Video Image Repair System',
        'page_subtitle': 'Upload a video to enhance its quality using our deep learning model',
        
        # Model settings
        'model_settings': 'Model Settings',
        'model_checkpoint': 'Model Checkpoint Path',
        'model_loaded': 'Model loaded successfully!',
        'using_untrained': 'Using untrained model - checkpoint not found',
        'using_device': 'Using device:',
        
        # Processing parameters
        'processing_params': 'Processing Parameters',
        'sequence_length': 'Sequence Length',
        'sequence_length_help': 'Number of frames to process at once',
        'frame_overlap': 'Frame Overlap',
        'frame_overlap_help': 'Number of overlapping frames between sequences',
        
        # Upload section
        'upload_video': 'Upload Video',
        'choose_video': 'Choose a video file',
        'supported_formats': 'Supported formats: MP4, AVI, MOV, MKV, WEBM (Max size: 500MB)',
        
        # Processing status
        'processing': 'Processing',
        'extracting_frames': 'Extracting frames from video...',
        'extracted_frames': 'Extracted {} frames',
        'processing_frames': 'Processing frames with sequence length {} and overlap {}...',
        'creating_video': 'Creating output video...',
        'processing_complete': 'Video processing completed!',
        
        # Results section
        'results': 'Results',
        'original_video': 'Original Video',
        'enhanced_video': 'Enhanced Video',
        'download_enhanced': 'Download Enhanced Video',
        'process_another': 'Process Another Video',
        
        # Error messages
        'error': 'Error',
        'try_again': 'Try Again',
        'load_error': 'Error loading model: {}',
        'process_error': 'Error processing video: {}',
        
        # Footer
        'footer': 'Deep Learning Video Image Repair System © 2023'
    }
}