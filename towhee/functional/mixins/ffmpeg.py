
class FFMPEGMixin():
    """"""
    @classmethod
    def decode_video(cls, filepath, threading = 'AUTO', num_worker = 0):
        """
        threading = 'AUTO', 'SPLICE', 'FRAME'
        codec = long list available on ffmpeg site
        """
        import av

        def inner():
            input_video = av.open(filepath)
            in_stream = input_video.streams.video[0]
            in_stream.thread_type = threading
            in_stream.thread_count = num_worker
            for packet in input_video.demux(in_stream):
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
            input_audio = av.open(filepath)
            in_stream = input_audio.streams.audio[0]
            in_stream.thread_type = threading
            in_stream.thread_count = num_worker
            for packet in input_audio.demux(in_stream):
                if packet.dts is None:
                    continue
                for frame in packet.decode():
                    yield frame.to_ndarray(format = 'fltp')

        return cls.stream(inner())

    def encode_video(self, filepath, video_codec = 'mpeg4', threading = 'AUTO', fps = 30, num_worker = 0):
        import av

        output = av.open(filepath, mode='w')

        video_stream = output.add_stream(video_codec, rate = fps)
        video_stream.codec_context.thread_count = num_worker
        video_stream.codec_context.thread_type = threading
        for video in self:
            frame = av.VideoFrame.from_ndarray(video, format='rgb24')
            for packet in video_stream.encode(frame):
                output.mux(packet)

        for packet in video_stream.encode():
            output.mux(packet)

        output.close()

    def encode_audio(self, filepath, audio_codec = 'aac', threading = 'AUTO', sample_rate = 44100, num_worker = 0):
        import av

        output = av.open(filepath, mode='w')

        audio_stream = output.add_stream(audio_codec, sample_rate = sample_rate)
        audio_stream.codec_context.thread_count = num_worker
        audio_stream.codec_context.thread_type = threading
        for audio in self:
            audio_frame = av.AudioFrame.from_ndarray(audio, format='fltp')
            audio_frame.sample_rate = sample_rate
            for packet in audio_stream.encode(audio_frame):
                output.mux(packet)

        for packet in audio_stream.encode():
            output.mux(packet)

        output.close()

    def encode_audio_and_video(self, filepath, audio_codec = 'aac', video_codec = 'mpeg4',
                                            threading = 'AUTO', fps = '30', sample_rate = 44100,
                                            num_worker = 0):
        import av
        import fractions

        output = av.open(filepath, mode='w')

        audio_stream = output.add_stream(audio_codec, sample_rate = sample_rate)
        audio_stream.codec_context.thread_count = num_worker
        audio_stream.codec_context.thread_type = threading
        video_stream = output.add_stream(video_codec, rate = fps)
        video_stream.codec_context.thread_count = num_worker
        video_stream.codec_context.thread_type = threading

        for i, (audio, video) in enumerate(self):

            audio_frame = av.AudioFrame.from_ndarray(audio, format="fltp")
            audio_frame.sample_rate = sample_rate
            audio_frame.pts = i *2000
            audio_frame.time_base = fractions.Fraction(1, 4800)
            for packet in audio_stream.encode(audio_frame):
                output.mux(packet)
            video_frame = av.VideoFrame.from_ndarray(video, format='rgb24')
            video_frame.pts = i *2000
            video_frame.time_base = fractions.Fraction(1, 4800)
            for packet in video_stream.encode(video_frame):
                output.mux(packet)

        for packet in audio_stream.encode():
            output.mux(packet)

        output.close()
