import pyopencl as cl
from sys import stdout

def choose_platform():
    platforms = cl.get_platforms()
    num_platforms = len(platforms)
    if (num_platforms == 1):
        return platforms[0]
    else:
        print('Available platforms:')
        for i in range(num_platforms):
            print('\t[{0}] {1}'.format(i, platforms[i]))
        while (True):
            print('Choose platform: ', end='')
            stdout.flush()
            choice = int(input())
            if (choice >= 0 and choice < num_platforms):
                return platforms[choice]

# ask the user what device(s) to use and returns them in a list
def choose_device(platform):
    devices = platform.get_devices()
    num_devices = len(devices)
    if (num_devices == 1):
        return platforms[0]
    else:
        print('Available devices:')
        print('\t[{0}] {1}'.format('a', 'Use all devices.'))
        for i in range(num_devices):
            print('\t[{0}] {1}'.format(i, devices[i]))
        while (True):
            print('Choose device: ', end='')
            stdout.flush()
            choice = input()
            try:
                choice = int(choice)
                if (choice >= 0 and choice < num_devices):
                    return [devices[choice]]
            # this is executed if str to int conversion fails
            except ValueError:
                if (choice == 'a'):
                    return devices

