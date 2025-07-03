import struct

HEADER_LEN = 3
ENV_SHAPES_HEADER = [82772.0, 83273.0, 83774.0]
ENV_RESET_STATE_HEADER = [83744.0, 83774.0, 83876.0]
ENV_STEP_DATA_HEADER = [83775.0, 53776.0, 83727.0]
POLICY_ACTIONS_HEADER = [12782.0, 83783.0, 80784.0]
PROC_MESSAGE_SHAPES_HEADER = [63776.0, 83777.0, 83778.0]
STOP_MESSAGE_HEADER = [11781.0, 83782.0, 83983.0]


def pack_message(message_floats):
    return struct.pack("%sf" % len(message_floats), *message_floats)


def unpack_message(message_bytes):
    return list(struct.unpack("%sf" % (len(message_bytes) // 4), message_bytes))
