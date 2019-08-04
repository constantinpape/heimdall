from heimdall import view_container


def example(load=False):
    path = '/home/pape/Work/data/ilastik/mulastik/data/data.h5'
    view_container(path, load_into_memory=load)


if __name__ == '__main__':
    example(True)
