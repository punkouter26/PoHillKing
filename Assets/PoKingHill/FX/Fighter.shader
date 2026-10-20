// Fighter surface for PoKingHill (BodyLook): URP physically based lighting (main light with shadows, additional lights,
// Forward+ cluster loop, fog) plus four match-driven layers, all set per renderer through a MaterialPropertyBlock:
//   _RimColor   fresnel rim in the seat colour (alpha = strength); BodyLook turns it red while the fighter is at the edge
//   _Wear       0..1 scratches and grime that build up from the contacts a body part took this match (object space,
//               so they stay on the part as it moves)
//   _Wet        0..1 darker, glossier surface after the sea; dries over about 20 s
//   _Flash      0..1 a short hot glow on a part that was just hit
// _BaseMap / _BumpMap are optional (the scanned humans have them, the G1 parts are plain colours).
Shader "PoKingHill/Fighter"
{
    Properties
    {
        _BaseColor ("Base colour", Color) = (0.5, 0.5, 0.5, 1)
        _BaseMap ("Base map", 2D) = "white" {}
        [Normal] _BumpMap ("Normal map", 2D) = "bump" {}
        _BumpScale ("Normal scale", Float) = 1
        _Metallic ("Metallic", Range(0, 1)) = 0.5
        _Smoothness ("Smoothness", Range(0, 1)) = 0.5
        [HDR] _EmissionColor ("Emission", Color) = (0, 0, 0, 1)
        _RimColor ("Rim colour (a = strength)", Color) = (1, 1, 1, 0)
        _RimPower ("Rim power", Range(0.5, 8)) = 3
        _Wear ("Wear", Range(0, 1)) = 0
        _Wet ("Wet", Range(0, 1)) = 0
        _Flash ("Hit flash", Range(0, 1)) = 0
        [Enum(UnityEngine.Rendering.CullMode)] _Cull ("Cull", Float) = 2
    }
    SubShader
    {
        Tags { "RenderType" = "Opaque" "Queue" = "Geometry" "RenderPipeline" = "UniversalPipeline" }

        HLSLINCLUDE
        #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
        CBUFFER_START(UnityPerMaterial)
            float4 _BaseMap_ST;
            half4 _BaseColor, _EmissionColor, _RimColor;
            half _BumpScale, _Metallic, _Smoothness, _RimPower, _Wear, _Wet, _Flash, _Cull;
        CBUFFER_END
        TEXTURE2D(_BaseMap); SAMPLER(sampler_BaseMap);
        TEXTURE2D(_BumpMap); SAMPLER(sampler_BumpMap);
        ENDHLSL

        Pass
        {
            Name "ForwardLit"
            Tags { "LightMode" = "UniversalForward" }
            Cull [_Cull]
            HLSLPROGRAM
            #pragma target 3.0
            #pragma vertex vert
            #pragma fragment frag
            // multi_compile, not shader_feature: BodyLook enables it on materials made at runtime (a build strips unused features)
            #pragma multi_compile_local _ _NORMALMAP
            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS _MAIN_LIGHT_SHADOWS_CASCADE _MAIN_LIGHT_SHADOWS_SCREEN
            #pragma multi_compile _ _ADDITIONAL_LIGHTS_VERTEX _ADDITIONAL_LIGHTS
            #pragma multi_compile_fragment _ _ADDITIONAL_LIGHT_SHADOWS
            #pragma multi_compile_fragment _ _SHADOWS_SOFT _SHADOWS_SOFT_LOW _SHADOWS_SOFT_MEDIUM _SHADOWS_SOFT_HIGH
            #pragma multi_compile _ _CLUSTER_LIGHT_LOOP
            #pragma multi_compile_fog
            #pragma multi_compile_instancing
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"

            struct Attributes { float4 positionOS : POSITION; float3 normalOS : NORMAL; float4 tangentOS : TANGENT; float2 uv : TEXCOORD0; UNITY_VERTEX_INPUT_INSTANCE_ID };
            struct Varyings
            {
                float4 positionCS : SV_POSITION; float3 positionWS : TEXCOORD0; float3 normalWS : TEXCOORD1; float4 tangentWS : TEXCOORD2;
                float2 uv : TEXCOORD3; float3 positionOS : TEXCOORD4; half fog : TEXCOORD5;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            Varyings vert(Attributes v)
            {
                Varyings o = (Varyings)0;
                UNITY_SETUP_INSTANCE_ID(v); UNITY_TRANSFER_INSTANCE_ID(v, o);
                VertexPositionInputs p = GetVertexPositionInputs(v.positionOS.xyz);
                VertexNormalInputs n = GetVertexNormalInputs(v.normalOS, v.tangentOS);
                o.positionCS = p.positionCS; o.positionWS = p.positionWS; o.normalWS = n.normalWS;
                o.tangentWS = float4(n.tangentWS, v.tangentOS.w * GetOddNegativeScale());
                o.uv = TRANSFORM_TEX(v.uv, _BaseMap); o.positionOS = v.positionOS.xyz; o.fog = ComputeFogFactor(p.positionCS.z);
                return o;
            }

            float Hash(float2 p) { p = frac(p * float2(123.34, 456.21)); p += dot(p, p + 45.32); return frac(p.x * p.y); }

            // Short straight scratches in three directions, more of them as wear grows; plus a soft grime mask.
            void Scratches(float3 p, half wear, out half scratch, out half grime)
            {
                scratch = 0;
                const float3 dirs[3] = { float3(0.80, 0.55, 0.24), float3(-0.30, 0.90, -0.31), float3(0.45, -0.20, 0.87) };
                [unroll] for (int i = 0; i < 3; i++)
                {
                    float3 a = dirs[i], b = normalize(cross(a, float3(0.31, 0.47, 0.83)));
                    float across = dot(p, b) * 110.0, along = dot(p, a) * 14.0;
                    float cell = Hash(float2(floor(across), floor(along) + i * 17.0));
                    float stroke = 1.0 - saturate(abs(frac(across) - 0.5) * 6.0);
                    // fade lines out once they get thinner than a pixel (far away), or they alias into glitter
                    float aa = saturate(1.5 - fwidth(across) * 3.0);
                    scratch = max(scratch, stroke * aa * step(cell, wear * 0.3));
                }
                grime = wear * (0.6 + 0.4 * sin(dot(p, float3(23.1, 17.3, 29.7))) * sin(dot(p, float3(-11.0, 31.0, 7.0))));
            }

            half4 frag(Varyings i) : SV_Target
            {
                UNITY_SETUP_INSTANCE_ID(i);
                half4 tex = SAMPLE_TEXTURE2D(_BaseMap, sampler_BaseMap, i.uv);
                half3 albedo = tex.rgb * _BaseColor.rgb;
                half metallic = _Metallic, smooth = _Smoothness;
                float3 n = normalize(i.normalWS);
                #if defined(_NORMALMAP)
                    half3 nTS = UnpackNormalScale(SAMPLE_TEXTURE2D(_BumpMap, sampler_BumpMap, i.uv), _BumpScale);
                    float3 bitan = i.tangentWS.w * cross(n, i.tangentWS.xyz);
                    n = normalize(TransformTangentToWorld(nTS, half3x3(i.tangentWS.xyz, bitan, n)));
                #endif

                half scratch = 0, grime = 0;
                if (_Wear > 0.001) Scratches(i.positionOS, _Wear, scratch, grime);
                albedo = lerp(albedo, albedo * 0.72, grime * 0.6);                    // grime darkens
                albedo = lerp(albedo, half3(0.62, 0.62, 0.64), scratch * 0.85);        // bare metal shows through
                metallic = lerp(metallic, 0.9, scratch); smooth = lerp(smooth * (1 - 0.35 * grime), 0.35, scratch);
                albedo *= lerp(1.0, 0.62, _Wet); smooth = lerp(smooth, 0.93, _Wet);   // wet: darker and glossy

                InputData d = (InputData)0;
                d.positionWS = i.positionWS; d.positionCS = i.positionCS; d.normalWS = n;
                d.viewDirectionWS = GetWorldSpaceNormalizeViewDir(i.positionWS);
                d.shadowCoord = TransformWorldToShadowCoord(i.positionWS);
                d.fogCoord = i.fog; d.bakedGI = SampleSH(n);
                d.normalizedScreenSpaceUV = GetNormalizedScreenSpaceUV(i.positionCS);
                d.shadowMask = half4(1, 1, 1, 1);

                SurfaceData s = (SurfaceData)0;
                s.albedo = albedo; s.metallic = metallic; s.smoothness = smooth; s.occlusion = 1; s.alpha = 1; s.specular = 0;
                half rim = pow(1.0 - saturate(dot(n, d.viewDirectionWS)), _RimPower) * _RimColor.a;
                s.emission = _EmissionColor.rgb + _RimColor.rgb * rim + half3(1.0, 0.62, 0.28) * (_Flash * _Flash * 4.0);
                half4 c = UniversalFragmentPBR(d, s);
                c.rgb = MixFog(c.rgb, i.fog);
                return half4(c.rgb, 1);
            }
            ENDHLSL
        }

        Pass
        {
            Name "ShadowCaster"
            Tags { "LightMode" = "ShadowCaster" }
            ZWrite On ZTest LEqual ColorMask 0 Cull [_Cull]
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_instancing
            #pragma multi_compile_vertex _ _CASTING_PUNCTUAL_LIGHT_SHADOW
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Shadows.hlsl"
            float3 _LightDirection, _LightPosition;
            struct Attributes { float4 positionOS : POSITION; float3 normalOS : NORMAL; UNITY_VERTEX_INPUT_INSTANCE_ID };
            float4 vert(Attributes v) : SV_POSITION
            {
                UNITY_SETUP_INSTANCE_ID(v);
                float3 p = TransformObjectToWorld(v.positionOS.xyz), n = TransformObjectToWorldNormal(v.normalOS);
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
            #pragma multi_compile_instancing
            struct Attributes { float4 positionOS : POSITION; UNITY_VERTEX_INPUT_INSTANCE_ID };
            float4 vert(Attributes v) : SV_POSITION { UNITY_SETUP_INSTANCE_ID(v); return TransformObjectToHClip(v.positionOS.xyz); }
            half frag() : SV_Target { return 0; }
            ENDHLSL
        }

        Pass
        {
            Name "DepthNormals"
            Tags { "LightMode" = "DepthNormals" }
            ZWrite On
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_instancing
            struct Attributes { float4 positionOS : POSITION; float3 normalOS : NORMAL; UNITY_VERTEX_INPUT_INSTANCE_ID };
            struct Varyings { float4 positionCS : SV_POSITION; float3 normalWS : TEXCOORD0; };
            Varyings vert(Attributes v) { UNITY_SETUP_INSTANCE_ID(v); Varyings o; o.positionCS = TransformObjectToHClip(v.positionOS.xyz); o.normalWS = TransformObjectToWorldNormal(v.normalOS); return o; }
            half4 frag(Varyings i) : SV_Target { return half4(NormalizeNormalPerPixel(i.normalWS), 0); }
            ENDHLSL
        }
    }
    FallBack "Hidden/Universal Render Pipeline/FallbackError"
}
