#include "interactions.h"

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{
    
    thrust::uniform_real_distribution<float> uniformDistribution{ 0.0f, 1.0f };
    // Use to stocastically determine between reflective, refractive, and diffuse surfaces
    float sampleProb = uniformDistribution(rng);

    if (sampleProb < m.hasRefractive) { // refractive
        float cosT = - glm::dot(pathSegment.ray.direction, normal);
        float sinT = sqrtf(1 - cosT * cosT);

        bool isEntering = cosT > 0.0f;
        float indexOfRefraction = isEntering ? (1.0f / m.indexOfRefraction) : m.indexOfRefraction;
    
        float reflectance0 = (1.0f - indexOfRefraction) / (1.0f + indexOfRefraction);
        reflectance0 *= reflectance0;
        float reflectChance = reflectance0 + (1.0f - reflectance0) * powf(1.0f - cosT, 5.0f);
    

        // Schlick's approximation to determine whether to reflect or refract
        float r0 = (1 - indexOfRefraction) / (1 + indexOfRefraction);
        r0 = r0 * r0;
        float schlickProbability = r0 + (1 - r0) * std::pow((1 - cosT), 5);

        // Total reflection
        if ((indexOfRefraction * sinT > 1) || (uniformDistribution(rng) < schlickProbability)) {
            pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
        }
        else {
            // glm::refract() implements Snell's law
            pathSegment.ray.direction = glm::refract(pathSegment.ray.direction, normal, indexOfRefraction);
        }
    }
    else if (sampleProb < m.hasReflective) { //reflective
        pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
    }
    else { // diffuse via cosine-weighted hemisphere
        pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
    }
    pathSegment.ray.origin = intersect + 0.0001f * pathSegment.ray.direction;
}
