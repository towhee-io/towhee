from towhee import pipe, AutoPipes, AutoConfig

@AutoConfig.register
class MyConfig:
    """
    For UT
    """
    def __init__(self):
        self.param = 10

@AutoPipes.register
def pipeline(config):
    """
    For UT
    """
    return (
        pipe.input('num')
        .map('num', 'res', lambda x: x + config.param)
        .output('res')
    )


