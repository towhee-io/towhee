class FFMPEGMixin():
    

    @classmethod
    def decode_video(cls, filepath, threading = 'AUTO', num_worker = 0):
        """
        threading = 'AUTO', 'SPLICE', 'FRAME'
        codec = long list available on ffmpeg site
        """
        import av

        def inner():
            input = av.open(filepath)
            in_stream = input.streams.video[0]
            in_stream.thread_type = threading
            in_stream.thread_count = num_worker
            for packet in input.demux(in_stream):
                if packet.dts is None:
                    continue
                for frame in packet.decode():
                    yield frame.to_ndarray(format='rgb24')

        return cls.stream(inner())
        
    @classmethod    
    def decode_audio(cls, filepath, threading = 'AUTO', num_worker = 0):
        """
        threading = 'AUTO', 'SPLICE', 'FRAME'
        codec = long list available on ffmpeg site
        """
        import av

        def inner():
            input = av.open(filepath)
            in_stream = input.streams.audio[0]
            in_stream.thread_type = threading
            in_stream.thread_count = num_worker
            for packet in input.demux(in_stream):
                if packet.dts is None:
                    continue
                for frame in packet.decode():
                    print(frame.sample_rate)
                    yield frame.to_ndarray()
        return cls.stream(inner())

    def encode_video(self, filepath, video_codec = 'mpeg4', threading = 'AUTO', rate = 30, num_worker = 0):
        import av

        output = av.open(filepath, mode='w')
        
        video_stream = output.add_stream(video_codec, rate = rate)
        video_stream.codec_context.thread_count = num_worker
        video_stream.codec_context.thread_type = threading
        for video in self:
            frame = av.VideoFrame.from_ndarray(video, format="rgb24")
            for packet in video_stream.encode(frame):
                output.mux(packet)

        for packet in video_stream.encode():
            output.mux(packet)
    
        output.close()

    def encode_audio(self, filepath, video_codec = 'mpeg4', threading = 'AUTO', sample_rate = 30, num_worker = 0):
       
        audio_stream = output.add_stream(audio_codec, rate = 44100)
        audio_stream.codec_context.thread_count = num_worker
        audio_stream.codec_context.thread_type = threading
        for video, audio in self:
            video_frame = av.VideoFrame.from_ndarray(video, format="rgb24")
            audio_frame = av.AudioFrame.from_ndarray(audio, format="fltp" )
            for packet in video_stream.encode(video_frame):
                output.mux(packet)
            for packet in audio_stream.encode(audio_frame):
                output.mux(packet)

        