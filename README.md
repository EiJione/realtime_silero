
# Voice Activity Detection using Silero VAD

This program utilizes the Silero Voice Activity Detection (VAD) model to analyze audio streams from PCM files to detect voice activity. It supports mono and stereo audio inputs and requires specifying the sample rate and channel depth.

## Requirements

- docker

## Compilation

Compile the program with docker support enabled. 

```bash
docker build -t sileroimg .
```

## Usage

Run the compiled application with the following command-line options:

```bash
docker run sileroimg -i /root/data/output.pcm -r 48000 -d 2
```

### Options

- `-i` Path to the input PCM file.
- `-r` Sample rate of the PCM file (default is 16000 Hz).
- `-d` Channel depth (1 for mono, 2 for stereo, default is 1).


## Output

The program outputs the index of each processed frame and whether voice is detected in that frame.

## Error Handling

- If the PCM file path is not provided or the file cannot be opened, the program will terminate with an appropriate error message.
- Errors during reading the PCM file are handled gracefully, indicating if the end of the file is reached or if a read error occurs.

## Contributing

Contributions to improve the code or extend the functionality are welcome. Please submit a pull request or open an issue to discuss your ideas.
