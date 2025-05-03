#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}


__device__ glm::vec2 generateStratifiedSample(glm::vec2 uniform, int index, int totalGridCells, bool useStratification) {
	if (!useStratification) return uniform;

	int gridResolution = static_cast<int>(sqrtf(static_cast<float>(totalGridCells)));
	float cellDimension = 1.0f / gridResolution;

	if (index >= gridResolution * gridResolution) {
		// If index exceeds the number of grid cells, fallback to uniform sampling
		return uniform;
	}

	int gridY = index / gridResolution;
	int gridX = index % gridResolution;

	glm::vec2 cellIndex = glm::vec2(gridX, gridY);
	return (cellIndex + uniform) * cellDimension;
}

__device__ glm::vec2 polarTransform(const glm::vec2 squareCoords) {
	glm::vec2 shiftedCoords = 2.0f * squareCoords - glm::vec2{ 1.0f };
	if (shiftedCoords.x == 0.0f && shiftedCoords.y == 0.0f)
		return { 0, 0 };

	float radius, angle;

	if (fabsf(shiftedCoords.x) > fabsf(shiftedCoords.y)) {
		radius = shiftedCoords.x;
		angle = PI_OVER_FOUR * shiftedCoords.y / shiftedCoords.x;
	} else {
		radius = shiftedCoords.y;
		angle = PI_OVER_TWO - PI_OVER_FOUR * shiftedCoords.x / shiftedCoords.y;
	}

	return radius * glm::vec2{ cosf(angle), sinf(angle) };
}

// Kernel that computes the condition buffer and partial image
__global__ void updateActiveMaskAndAccumulateColor(
    PathSegment* pathSegments, int totalPaths, bool* activeMask, glm::vec3* finalImage
) {
    int pixelIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pixelIdx >= totalPaths) return;

    PathSegment currentPath = pathSegments[pixelIdx];

    if (currentPath.remainingBounces <= 0) {
        activeMask[pixelIdx] = true;
        finalImage[currentPath.pixelIndex] += currentPath.color;
    } else {
        activeMask[pixelIdx] = false;
    }
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...
static bool* dev_boolBuffer = NULL;

// SORT MATERIALS
thrust::device_ptr<int> pathsMaterial = nullptr;
thrust::device_ptr<int> intersectionsMaterial = nullptr;

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need
    cudaMalloc(&dev_boolBuffer, pixelcount * sizeof(bool));

    // Sort Materials
    int* pathsMaterial_ptr;
    int* intersectionsMaterial_ptr;
    
    cudaMalloc(&pathsMaterial_ptr, pixelcount * sizeof(int));
    cudaMemset(pathsMaterial_ptr, 0, pixelcount * sizeof(int));
    
    cudaMalloc(&intersectionsMaterial_ptr, pixelcount * sizeof(int));
    cudaMemset(intersectionsMaterial_ptr, 0, pixelcount * sizeof(int));
    
    pathsMaterial = thrust::device_pointer_cast(pathsMaterial_ptr);
    intersectionsMaterial = thrust::device_pointer_cast(intersectionsMaterial_ptr);    

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
    cudaFree(dev_boolBuffer);

    // sort materials
    cudaFree(pathsMaterial.get());
	cudaFree(intersectionsMaterial.get());


    checkCUDAError("pathtraceFree");
}



/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, bool hasDoF, bool hasStratified, int numCells)
{
    
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, pathSegments[index].remainingBounces);

        // Generate two seudo random numbers from unifrom distribution
        thrust::uniform_real_distribution<float> sampleDistribution(-0.5f, 0.5f);
        float jitterX = sampleDistribution(rng);
        float jitterY = sampleDistribution(rng);

        // Add ray jitter for antialiasing
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + jitterX - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + jitterY - (float)cam.resolution.y * 0.5f)
        );
      
        if (hasDoF) {
            thrust::default_random_engine randomGen = makeSeededRandomEngine(iter, index, -2);
            thrust::uniform_real_distribution<float> randDist(0, 1);

            glm::vec2 aperturePoint = cam.aperture * polarTransform(
                generateStratifiedSample(glm::vec2{ randDist(randomGen), randDist(randomGen) }, iter, numCells, hasStratified)
            );

            float rayViewDot = glm::dot(segment.ray.direction, cam.view);
            float focalT = cam.focalDistance / rayViewDot;

            glm::vec3 focalPoint = segment.ray.origin + focalT * segment.ray.direction;
            segment.ray.origin += aperturePoint.x * cam.right + aperturePoint.y * cam.up;
            segment.ray.direction = glm::normalize(focalPoint - segment.ray.origin);

        }

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;

        }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections, 
    bool sortMaterials,
    thrust::device_ptr<int> pathsMaterial,
	thrust::device_ptr<int> intersectionsMaterial)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            // TODO: add more intersection tests here... triangle? metaball? CSG?

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
        }
        if (sortMaterials)
        {
            pathsMaterial[path_index] = intersections[path_index].materialId;
		    intersectionsMaterial[path_index] = intersections[path_index].materialId;
        }
    }
}

// BSDF shading kernel
__global__ void shadeFakeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        PathSegment& pathSegment = pathSegments[idx];
        if (pathSegment.remainingBounces <= 0) return;

        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material object is light, "light" the ray and set bounces to 0
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
                pathSegment.remainingBounces = 0;
            }
            else {
                // Set up the RNG
                thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegment.remainingBounces);
                // Calculate point of intersection given the Ray (origin and direction) and the scalar distance to the instersection
                glm::vec3 pointOfIntersection = getPointOnRay(pathSegment.ray, intersection.t);

                // If the material is not refractive, multiply the color by the material color
                if (!material.hasRefractive){
                    pathSegment.color *= materialColor;        
                }

                // Funcation that calculates the new ray direction based on material property
                scatterRay(pathSegment, pointOfIntersection, intersection.surfaceNormal, material, rng);

                --pathSegment.remainingBounces;
                

            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            pathSegments[idx].color = glm::vec3(0.0f);
            pathSegment.remainingBounces = 0;

        }
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}


void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    if (guiData == nullptr){
        generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths, false, false, 225);
    }
    else {
        generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths, guiData->DoF, guiData->Stratified, guiData->StratNumCells);

    }
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    unsigned int num_paths = pixelcount;

    // Wrapping device pointers in thrust device pointers for stream compaction
	thrust::device_ptr<PathSegment> thrust_pathsegments(dev_paths);
	thrust::device_ptr<PathSegment> thrust_pathsegments_end(dev_path_end);
	thrust::device_ptr<bool> thrust_conditionalBuffer(dev_boolBuffer);

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks
    dim3 numBlocks1d { (num_paths + blockSize1d - 1) / blockSize1d };

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing with sort materials
        computeIntersections<<<numBlocks1d, blockSize1d>>> (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_intersections,
            guiData->SortMaterials,
            pathsMaterial,
            intersectionsMaterial
        );
        if(guiData->SortMaterials)
        {
            thrust::sort_by_key(thrust::device, pathsMaterial, pathsMaterial + num_paths, dev_paths);
            thrust::sort_by_key(thrust::device, intersectionsMaterial, intersectionsMaterial + num_paths, dev_intersections);    
        }
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;


        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading  
        // path segments that have been reshuffled to be contiguous in memory.

        shadeFakeMaterial<<<numBlocks1d, blockSize1d>>>(
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials);
        cudaDeviceSynchronize();

        // Calculate the condition buffer and apply stream compaction via thrust::remove_if
        if (guiData == nullptr || guiData->StreamCompaction) {
			updateActiveMaskAndAccumulateColor<<<numBlocks1d, blockSize1d>>>(
				dev_paths, num_paths, dev_boolBuffer, dev_image);

			// Using thrust to remove inactive paths
			thrust_pathsegments_end = thrust::remove_if(
				thrust_pathsegments, thrust_pathsegments + num_paths, thrust_conditionalBuffer, thrust::identity<bool>());
			cudaDeviceSynchronize();

			num_paths = thrust_pathsegments_end - thrust_pathsegments;
			numBlocks1d = dim3{ (num_paths + blockSize1d - 1) / blockSize1d };

			if (num_paths == 0){
                iterationComplete = true;
            }
		}

        if(depth == traceDepth) iterationComplete = true; // added conditional to stop iterations based on traceDepth

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }


    if (guiData == nullptr || guiData->StreamCompaction) {
		if (num_paths) {
			finalGather<<<numBlocks1d, blockSize1d>>>(num_paths, dev_image, dev_paths);
			checkCUDAError("finalGather");
		}
	}
	else {
		finalGather<<<numBlocks1d, blockSize1d>>>(pixelcount, dev_image, dev_paths);
		checkCUDAError("finalGather");
	}

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
