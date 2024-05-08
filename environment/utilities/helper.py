def find_last_queue_slot_index(queue):
    for i, slot in enumerate(queue):
        if slot == (0, -1):
            return i
    return -1


def is_core_free(queue):
    if queue[0] == (0, -1):
        return True
    else:
        return False


def queue_shift_left(queue):
    queue.pop(0)
    queue.append((0, -1))
