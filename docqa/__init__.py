from hashlib import sha1


class NoIndexError(Exception):
    pass


class RetrieverNotInitialized(Exception):
    pass

def file_sha1(filename):
    h = sha1()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(filename, "rb", buffering=0) as f:
        while n := f.readinto(mv):
            h.update(mv[:n])

    return str(h.hexdigest())
