from tap import Tap
import torch

class SimpleArgumentParser(Tap):
    name: str  # Your name
    language: str = 'Python'  # Programming language
    package: str = 'Tap'  # Package name
    stars: int  # Number of stars
    max_stars: int = 5  # Maximum stars
    work_dir = "./workdir"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = SimpleArgumentParser().parse_args()
