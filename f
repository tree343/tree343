implementing "../texture.slang";

public struct SharedTexture3DBuffer<T>
    where T : IFloat
{
    public StructuredBuffer<T> buffer;

    T getBufferValue(SharedTexture3D tex, uint3 voxelCoord)
    {
        uint3 texSize = tex.size;
        uint offset = tex.offset;
        uint index = voxelCoord.z * texSize.y * texSize.x +
                     voxelCoord.y * texSize.x +
                     voxelCoord.x;
        return buffer[offset + index];
    }

    /**
     * Sample the texture at the given uvw coordinates using nearest neighbor interpolation.
     * @param tex The texture to sample.
     * @param uvw The uvw coordinates to sample the texture at.
     * @return The sampled value.
     */
    public T pointSample(SharedTexture3D tex, float3 uvw)
    {
        // TODO: Student implementation starts here.

        // return T(0);

        // STEP 1: Clamp UVW to valid [0, 1] range
        // This prevents reading outside the texture bounds
        uvw = clamp(uvw, float3(0.0), float3(1.0));

        // STEP 2: Convert normalized coordinates to voxel coordinates
        // UVW of (0, 0, 0) maps to voxel (0, 0, 0)
        // UVW of (1, 1, 1) maps to the last voxel (size - 1)
        float3 voxelCoordF = uvw * float3(tex.size - uint3(1, 1, 1));

        // STEP 3: Round to the nearest integer voxel coordinate
        // This is the "nearest neighbor" part — we pick the closest voxel
        uint3 voxelCoord = uint3(round(voxelCoordF));

        // STEP 4: Clamp to ensure we're within valid bounds (safety check)
        voxelCoord = clamp(
            voxelCoord,
            uint3(0, 0, 0),
            tex.size - uint3(1, 1, 1)
        );

        // STEP 5: Return the voxel value
        return getBufferValue(tex, voxelCoord);



        // TODO: Student implementation ends here.
    }

    /**
     * Sample the texture at the given uvw coordinates using trilinear interpolation.
     * @param tex The texture to sample.
     * @param uvw The uvw coordinates to sample the texture at.
     * @return The sampled value.
    **/
    public T trilinearSample(SharedTexture3D tex, float3 uvw)
    {
        // TODO: Student implementation starts here.

        //return T(0);
        float3 voxel_size = float3(1.0) / tex.size;
        float3 voxel_coords = uvw / voxel_size - float3(0.5);
        uint3 v000 = (uint3)voxel_coords;
        float3 frac = voxel_coords - v000;
        uint3 v111 = min(v000 + uint3(1), tex.size - uint3(1));

        T c000 = getBufferValue(tex, uint3(v000.x, v000.y, v000.z)); 
        T c100 = getBufferValue(tex, uint3(v111.x, v000.y, v000.z)); 
        T c010 = getBufferValue(tex, uint3(v000.x, v111.y, v000.z)); 
        T c110 = getBufferValue(tex, uint3(v111.x, v111.y, v000.z)); 
        T c001 = getBufferValue(tex, uint3(v000.x, v000.y, v111.z)); 
        T c101 = getBufferValue(tex, uint3(v111.x, v000.y, v111.z)); 
        T c011 = getBufferValue(tex, uint3(v000.x, v111.y, v111.z));
        T c111 = getBufferValue(tex, uint3(v111.x, v111.y, v111.z));

        T c00 = lerpT(c000, c100, frac.x);
        T c10 = lerpT(c010, c110, frac.x); 
        T c01 = lerpT(c001, c101, frac.x); 
        T c11 = lerpT(c011, c111, frac.x); 

        T c0 = lerpT(c00, c10, frac.y);
        T c1 = lerpT(c01, c11, frac.y); 

        return lerpT(c0, c1, frac.z);

        // TODO: Student implementation ends here.
    }

    public T lerpT(T a, T b, float t) {
        return a + (b - a) * T(t);
    }

}

public struct SharedTexture3D
{
    public uint3 size;
    public uint offset;
}

