class FFMPEGMixin():
    
    def decode_audio(self):
        pass

    @classmethod
    def decode_video(cls, filepath, threading = 'AUTO', num_worker = 0):
        """
        threading = 'AUTO', 'SPLICE', 'FRAME'
        codec = long list available on ffmpeg site
        """
        import av

        def inner():
            container = av.open(filepath)
            container.streams.video[0].thread_type = threading
            container.streams.video[0].thread_count = num_worker
            for frame in container.decode(video=0):
                image = frame.to_ndarray(format='rgb24')
                yield image
            container.close()
            
        return cls.stream(inner())
        
                
    def encode_audio(self):
        pass

    def encode_video(self, filepath, codec = 'mpeg4', threading = 'AUTO', rate = 30, num_worker = 0):
        import av
        import numpy as np
        print('ran here')
        container = av.open(filepath, mode='w')
        stream = container.add_stream(codec, rate = rate)
        stream.codec_context.thread_count = num_worker
        stream.codec_context.thread_type = threading
        for i, img in enumerate(self):
            # print(i)
            # img = np.round(255 * img).astype(np.uint8)
            # img = np.clip(img, 0, 255)
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)

        # Close the file
        container.close()

        