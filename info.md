implementing "../renderer.slang";

import math;
import texture;

public static uint MAX_STEPS = 1024;
public static float EPSILON = 1e-6;

public static float STEP_SIZE = 0.01;
public static float DENSITY_SCALE = 5.0;

/**
 * Perform ray marching through the volume and accumulate color.
 * @param uv The normalized uv coordinate from (0, 0) in the bottom-left to (1, 1) in the top-right.
 * @param uniforms The renderer uniform containing camera, sampling, and scene information.
 * @param tMax The maximum t value for ray marching (usually the nearest primitive intersection t).
 * @param primColor The color of the nearest primitive intersection.
 * @return The accumulated color after ray marching through the volume.
 */
public float4 volumeSample(float2 uv, RendererUniform uniforms, float tMax, float4 primColor)
{
    // TODO: Student implementation starts here.
    
    Ray ray = uniforms.camera.generateRay(uv);

    DiffVolume volume = uniforms.volume;
    BoundingBox volumeBound = volume.bound;

    Ray localRay = ray.transform(volume.invModelMatrix);

    float2 tHit = volumeBound.hit(localRay);

    if (tHit.x > tHit.y || tHit.x > tMax)
        return primColor;

    float tStart = max(tHit.x, 0.0);
    float tEnd   = min(tHit.y, tMax);

    if (tStart >= tEnd)
        return primColor;

    float4 accumulatedColor = float4(0.0);
    float transmittance = 1.0;

    for (uint step = 0; step < MAX_STEPS; step++)
    {
        float t = tStart + float(step) * STEP_SIZE;

        if (t >= tEnd || transmittance < EPSILON)
            break;

        float3 pos = localRay.origin + t * localRay.direction;

        float3 uvw = (pos - volumeBound.pMin) /
                     (volumeBound.pMax - volumeBound.pMin);

        if (any(uvw < float3(0.0)) || any(uvw > float3(1.0)))
            continue;

        float4 sampleVal = sampleTrilinear(volume.tex, uvw);

        float3 emission = sampleVal.xyz;
        float density = sampleVal.w * DENSITY_SCALE * STEP_SIZE;

        float alpha = 1.0 - exp(-density);

        accumulatedColor.xyz += transmittance * alpha * emission;

        transmittance *= (1.0 - alpha);
    }

    accumulatedColor.xyz += transmittance * primColor.xyz;
    accumulatedColor.w = 1.0 - transmittance + transmittance * primColor.w;

    return accumulatedColor;
    
    // TODO: Student implementation ends here.
}

/**
 * Backward pass for volumetric ray marching at a single uv.
 * @param uv The normalized uv coordinate from (0, 0) in the bottom-left to (1, 1) in the top-right.
 * @param uniforms The renderer uniform containing camera, sampling, and scene information.
 * @param outGrad The output gradient w.r.t. the sampled color.
 */
public void volumeSampleBwd(float2 uv, RendererUniform uniforms, float4 outGrad)
{
    // TODO: Student implementation starts here.

   Ray ray = uniforms.camera.generateRay(uv);

    DiffVolume volume = uniforms.volume;
    BoundingBox volumeBound = volume.bound;

    Ray localRay = ray.transform(volume.invModelMatrix);

    float2 tHit = volumeBound.hit(localRay);

    if (tHit.x > tHit.y)
        return;

    float tStart = max(tHit.x, 0.0);
    float tEnd   = tHit.y;

    if (tStart >= tEnd)
        return;

    float transmittance = 1.0;

    for (uint step = 0; step < MAX_STEPS; step++)
    {
        float t = tStart + float(step) * STEP_SIZE;

        if (t >= tEnd || transmittance < EPSILON)
            break;

        float3 pos = localRay.origin + t * localRay.direction;

        float3 uvw = (pos - volumeBound.pMin) /
                     (volumeBound.pMax - volumeBound.pMin);

        if (any(uvw < float3(0.0)) || any(uvw > float3(1.0)))
            continue;

        float4 sampleVal = sampleTrilinear(volume.tex, uvw);

        float density = sampleVal.w * DENSITY_SCALE * STEP_SIZE;
        float alpha = 1.0 - exp(-density);

        float4 dSample = float4(
            transmittance * alpha * outGrad.x,
            transmittance * alpha * outGrad.y,
            transmittance * alpha * outGrad.z,
            0.0
        );

        sampleTrilinear_bwd<float, 4>(
            DifferentialPtrPair<DiffTexture3D<float, 4>>(volume.tex, volume.dTex),
            uvw,
            dSample
        );

        transmittance *= (1.0 - alpha);
    }

    // TODO: Student implementation ends here.
}

/**
 * Ray march through the volume and accumulate color.
 * @param ray The ray in local space.
 * @param volumeBound The bounding box of the volume in local space.
 * @param maxSteps The maximum number of marching steps.
 * @param stepSize The size of each marching step.
 * @param densityScale The scale factor for the volume density (alpha channel).
 * @param backgroundColor The background color behind the volume
                          (this can be the ambient color if there's no primitive behind the volume, or the color of the primitive hit if there's one).
 * @param data The 3D texture storing the volume data (rgba).
 * @return The accumulated color after ray marching through the volume.
 */
[Differentiable]
public float4 accumulateColor<let N : uint>(
    no_diff Ray ray,
    no_diff BoundingBox volumeBound,
    no_diff uint maxSteps,
    no_diff float stepSize,
    no_diff float densityScale,
    no_diff float4 backgroundColor,
    DiffTexture3D<float, 4> data)
{
    // TODO: Student implementation starts here.

    float2 tHit = volumeBound.hit(ray);

    if (tHit.x > tHit.y)
        return backgroundColor;

    float tStart = max(tHit.x, 0.0);
    float tEnd   = tHit.y;

    if (tStart >= tEnd)
        return backgroundColor;

    float4 accumulatedColor = float4(0.0);
    float transmittance = 1.0;

    [ForceUnroll]
    for (uint step = 0; step < MAX_STEPS; step++)
    {
        if (step >= maxSteps)
            break;

        float t = tStart + float(step) * stepSize;

        if (t >= tEnd || transmittance < EPSILON)
            break;

        float3 pos = ray.origin + t * ray.direction;

        float3 uvw = (pos - volumeBound.pMin) /
                     (volumeBound.pMax - volumeBound.pMin);

        if (any(uvw < float3(0.0)) || any(uvw > float3(1.0)))
            continue;

        float4 sampleVal = sampleTrilinear(data, uvw);

        float3 emission = sampleVal.xyz;
        float density = sampleVal.w * densityScale * stepSize;

        float alpha = 1.0 - exp(-density);

        accumulatedColor.xyz += transmittance * alpha * emission;

        transmittance *= (1.0 - alpha);
    }

    accumulatedColor.xyz += transmittance * backgroundColor.xyz;
    accumulatedColor.w = 1.0 - transmittance + transmittance * backgroundColor.w;

    return accumulatedColor;

    // TODO: Student implementation ends here.
}
