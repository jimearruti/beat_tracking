import mirdata
from functions import beats_simple


def beatTracker(input_file):
    return beats_simple(input_file, downbeats=True)

def main(input_file):
    beats, downbeats = beatTracker(input_file)
    print("Beats:", beats)
    print("Downbeats:", downbeats)

if __name__ == "__main__":
    ballroom = mirdata.initialize('ballroom')
    file = ballroom.track('Media-105901').audio_path
    main(file)
