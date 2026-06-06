from .base_solution import Solution
from .triangle_embedding_solution import TriangleEmbeddingSolution
from .triangle_cielab_solution import TriangleCIELABSolution
from .triangle_mse_solution import TriangleMSESolution
from .triangle_solution import TriangleSolution

__all__ = [
	"Solution",
	"TriangleSolution",
	"TriangleCIELABSolution",
	"TriangleMSESolution",
	"TriangleEmbeddingSolution",
]
