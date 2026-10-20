// Banner cloth for the Mountain map (tools/make_mountain_map.py): URP physically based lighting, double sided, the
// seat colour from vertex colours times the emblem texture. The cloth sways in the wind MatchFx publishes as the global
// _KothWind (xyz = horizontal direction in world space, w = strength: about 0.3 calm, 1.3 in a Hazards gust). The
// mesh's u coordinate is the distance from the pole (0 at the pole), so the pole edge stays put. Render only.
Shader "PoKingHill/Banner"
{
    Properties
    {
        _BaseColor ("Tint", Color) = (1, 1, 1, 1)
        _BaseMap ("Emblem", 2D) = "white" {}
        _Smoothness ("Smoothness", Range(0, 1)) = 0.25
    }
    SubShader
    {
        Tags { "RenderType" = "Opaque" "Queue" = "Geometry" "RenderPipeline" = "UniversalPipeline" }
        Cull Off

        HLSLINCLUDE
        #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
        CBUFFER_START(UnityPerMaterial)
            float4 _BaseMap_ST; half4 _BaseColor; half _Smoothness;
        CBUFFER_END
        TEXTURE2D(_BaseMap); SAMPLER(sampler_BaseMap);
        float4 _KothWind;

        // Cloth: a wave travelling out along the flag plus a fast flutter, both growing away from the pole (u), the
        // whole flag pushed downwind and rippling a little up and down. f(u) is the offset along the flag's normal;
        // its slope df/du tilts the shading normal, so the folds show as moving light and dark bands from any side.
        void Cloth(inout float3 p, inout float3 n, float3 along, float u, float v)
        {
            float s = 0.15 + _KothWind.w, t = _Time.y, a = 0.03 + 0.10 * s;
            float ph1 = t * (2.5 + 3.5 * s) - u * 6.0 + p.x * 0.7 + p.z * 0.5, ph2 = t * (9.0 + 6.0 * s) - u * 15.0 + v * 4.0;
            float wave = sin(ph1) + 0.3 * sin(ph2);
            float slope = a * wave + u * a * (-6.0 * cos(ph1) - 4.5 * cos(ph2));
            p += n * u * a * wave + _KothWind.xyz * u * u * 0.12 * s + float3(0, 1, 0) * u * 0.25 * a * cos(ph1);
            n = normalize(n - along * slope);
        }
        float3 Sway(float3 p, float3 n, float3 along, float u, float v) { Cloth(p, n, along, u, v); return p; }
        ENDHLSL

        Pass
        {
            Name "ForwardLit"
            Tags { "LightMode" = "UniversalForward" }
            HLSLPROGRAM
            #pragma target 3.0
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS _MAIN_LIGHT_SHADOWS_CASCADE _MAIN_LIGHT_SHADOWS_SCREEN
            #pragma multi_compile _ _ADDITIONAL_LIGHTS_VERTEX _ADDITIONAL_LIGHTS
            #pragma multi_compile_fragment _ _SHADOWS_SOFT _SHADOWS_SOFT_LOW _SHADOWS_SOFT_MEDIUM _SHADOWS_SOFT_HIGH
            #pragma multi_compile _ _CLUSTER_LIGHT_LOOP
            #pragma multi_compile_fog
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"

            struct Attributes { float4 positionOS : POSITION; float3 normalOS : NORMAL; float4 tangentOS : TANGENT; float2 uv : TEXCOORD0; half4 color : COLOR; };
            struct Varyings { float4 positionCS : SV_POSITION; float3 positionWS : TEXCOORD0; float3 normalWS : TEXCOORD1; float2 uv : TEXCOORD2; half4 color : TEXCOORD3; half fog : TEXCOORD4; };

            Varyings vert(Attributes a)
            {
                Varyings o;
                float3 n = TransformObjectToWorldNormal(a.normalOS), p = TransformObjectToWorld(a.positionOS.xyz);
                Cloth(p, n, TransformObjectToWorldDir(a.tangentOS.xyz), a.uv.x, a.uv.y);       // tangent = along u, away from the pole
                o.positionWS = p; o.positionCS = TransformWorldToHClip(p); o.normalWS = n;
                o.uv = TRANSFORM_TEX(a.uv, _BaseMap); o.color = a.color; o.fog = ComputeFogFactor(o.positionCS.z);
                return o;
            }

            half4 frag(Varyings i, bool front : SV_IsFrontFace) : SV_Target
            {
                float3 n = normalize(i.normalWS) * (front ? 1 : -1);
                InputData d = (InputData)0;
                d.positionWS = i.positionWS; d.positionCS = i.positionCS; d.normalWS = n;
                d.viewDirectionWS = GetWorldSpaceNormalizeViewDir(i.positionWS);
                d.shadowCoord = TransformWorldToShadowCoord(i.positionWS); d.fogCoord = i.fog; d.bakedGI = SampleSH(n);
                d.normalizedScreenSpaceUV = GetNormalizedScreenSpaceUV(i.positionCS); d.shadowMask = half4(1, 1, 1, 1);
                SurfaceData s = (SurfaceData)0;
                s.albedo = SAMPLE_TEXTURE2D(_BaseMap, sampler_BaseMap, i.uv).rgb * _BaseColor.rgb * i.color.rgb;
                s.smoothness = _Smoothness; s.occlusion = 1; s.alpha = 1;
                half4 c = UniversalFragmentPBR(d, s);
                return half4(MixFog(c.rgb, i.fog), 1);
            }
            ENDHLSL
        }

        Pass
        {
            Name "ShadowCaster"
            Tags { "LightMode" = "ShadowCaster" }
            ZWrite On ZTest LEqual ColorMask 0
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_vertex _ _CASTING_PUNCTUAL_LIGHT_SHADOW
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Shadows.hlsl"
            float3 _LightDirection, _LightPosition;
            struct Attributes { float4 positionOS : POSITION; float3 normalOS : NORMAL; float4 tangentOS : TANGENT; float2 uv : TEXCOORD0; };
            float4 vert(Attributes a) : SV_POSITION
            {
                float3 n = TransformObjectToWorldNormal(a.normalOS);
                float3 p = Sway(TransformObjectToWorld(a.positionOS.xyz), n, TransformObjectToWorldDir(a.tangentOS.xyz), a.uv.x, a.uv.y);
                #if defined(_CASTING_PUNCTUAL_LIGHT_SHADOW)
                    float3 l = normalize(_LightPosition - p);
                #else
                    float3 l = _LightDirection;
                #endif
                float4 c = TransformWorldToHClip(ApplyShadowBias(p, n, l));
                #if UNITY_REVERSED_Z
                    c.z = min(c.z, UNITY_NEAR_CLIP_VALUE);
                #else
                    c.z = max(c.z, UNITY_NEAR_CLIP_VALUE);
                #endif
                return c;
            }
            half4 frag() : SV_Target { return 0; }
            ENDHLSL
        }

        Pass
        {
            Name "DepthOnly"
            Tags { "LightMode" = "DepthOnly" }
            ZWrite On ColorMask R
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            struct Attributes { float4 positionOS : POSITION; float3 normalOS : NORMAL; float4 tangentOS : TANGENT; float2 uv : TEXCOORD0; };
            float4 vert(Attributes a) : SV_POSITION
            {
                float3 n = TransformObjectToWorldNormal(a.normalOS);
                return TransformWorldToHClip(Sway(TransformObjectToWorld(a.positionOS.xyz), n, TransformObjectToWorldDir(a.tangentOS.xyz), a.uv.x, a.uv.y));
            }
            half frag() : SV_Target { return 0; }
            ENDHLSL
        }
    }
    FallBack "Hidden/Universal Render Pipeline/FallbackError"
}
