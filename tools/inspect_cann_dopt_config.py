"""Print the public protobuf schema shipped with the official DOPT tool."""
from dopt.notrain_onnx import notrain_pb2

def show(message, indent=''):
    print(indent+message.full_name)
    for field in message.fields:
        enum=[(v.name,v.number) for v in field.enum_type.values] if field.enum_type else None
        print(indent,field.name,'type',field.type,'repeated',field.label==3,'enum',enum)
    for nested in message.nested_types:
        show(nested,indent+'  ')

for message in notrain_pb2.DESCRIPTOR.message_types_by_name.values():
    show(message)
