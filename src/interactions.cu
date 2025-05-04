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
    PathSegment & segment,
    glm::vec3 hitPoint,
    glm::vec3 surfNormal,
    const Material &material,
    thrust::default_random_engine &engine)
{
    thrust::uniform_real_distribution<float> dist{ 0.0f, 1.0f };
    float randProb = dist(engine);

    if (randProb < material.hasRefractive) { // refractive case
        float cosIncidentAngle = -glm::dot(segment.ray.direction, surfNormal);
        float sinIncidnetAngle = sqrtf(1 - cosIncidentAngle * cosIncidentAngle);
        bool entering = cosIncidentAngle > 0.0f;
        float refIdx = entering ? (1.0f / material.indexOfRefraction) : material.indexOfRefraction;
    
        float baseReflectance = (1 - refIdx) / (1 + refIdx);
        baseReflectance = baseReflectance * baseReflectance;
        float schlickProb = baseReflectance + (1 - baseReflectance) * std::pow((1 - cosIncidentAngle), 5);

        if ((refIdx * sinIncidnetAngle > 1) || (dist(engine) < schlickProb)) {
            segment.ray.direction = glm::reflect(segment.ray.direction, surfNormal);
        } else {
            segment.ray.direction = glm::refract(segment.ray.direction, surfNormal, refIdx);
        }
    }
    else if (randProb < material.hasReflective) { // reflective case
        segment.ray.direction = glm::reflect(segment.ray.direction, surfNormal);
    }
    else { // diffuse via cosine-weighted hemisphere case
        segment.ray.direction = calculateRandomDirectionInHemisphere(surfNormal, engine);
    }
    segment.ray.origin = hitPoint + 0.0001f * segment.ray.direction;
}
