// Full-screen shockwave for PoKingHill (ShockwaveFeature): the camera image is displaced outward inside a ring that
// grows from the impact, with a faint bright rim. Inputs from Shockwave.Apply: _ShockCenter.xy = impact in viewport uv,
// _ShockParams = (radius, ring width, strength, aspect), all in units of the viewport height.
Shader "PoKingHill/Shockwave"
{
    SubShader
    {
        Tags { "RenderType" = "Opaque" "RenderPipeline" = "UniversalPipeline" }
        ZWrite Off ZTest Always Cull Off Blend Off

        Pass
        {
            Name "Shockwave"
            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment Frag
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.core/Runtime/Utilities/Blit.hlsl"

            float4 _ShockCenter, _ShockParams;

            half4 Frag(Varyings i) : SV_Target
            {
                float2 uv = i.texcoord;
                float2 d = uv - _ShockCenter.xy; d.x *= _ShockParams.w;           // aspect-correct distance
                float r = length(d), radius = _ShockParams.x, width = _ShockParams.y;
                float x = (r - radius) / width;                                    // -1..1 across the ring
                float ring = saturate(1 - x * x);
                float2 dir = r > 1e-4 ? d / r : 0; dir.x /= _ShockParams.w;
                float2 offs = dir * ring * ring * _ShockParams.z * sign(x + 1e-4) * -1;
                half4 c = SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, uv + offs);
                c.rgb += ring * ring * _ShockParams.z * 2;                         // a faint bright rim on the wave front
                return c;
            }
            ENDHLSL
        }
    }
}
