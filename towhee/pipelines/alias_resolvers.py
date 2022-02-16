#!/usr/bin/env python3
from towhee import ops
class AliasResolverBase:
    """
    Base class for alias resolvers
    """
    def resolve(self, name: str):
        pass

class LocalAliasResolver(AliasResolverBase):
    """
    Resolve aliases with locally with builtin rules
    """

    aliases = {
        'efficientnet-b3': ops.filip_halt.timm_image_embedding(model_name='efficientnet_b3'),
        'regnety-004': ops.filip_halt.timm_image_embedding(model_name = 'regnety-004')
    }

    def resolve(self, name: str):
        return LocalAliasResolver.aliases[name]

class RemoteAliasResolver(AliasResolverBase):
    """
    Resolve aliases from towhee hub
    """

    def resolve(self, name: str):
        pass

def get_resolver(name: str) -> AliasResolverBase:
    if name == 'local':
        return LocalAliasResolver()
    if name == 'remote':
        return RemoteAliasResolver()
    return None
