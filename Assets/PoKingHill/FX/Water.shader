// Rising-sea surface for PoKingHill (URP, transparent, no depth texture needed so it runs on the mobile renderer).
// Procedural ripples from a few moving sine sets perturb the normal; fresnel mixes deep and sky colours; a foam band
// sits where the water meets the hill, found analytically from the dome profile z = -k (r - R)^2 and the current
// level (_SeaLevel, set by MatchFx). _Storm darkens and roughens the sea as it nears the summit.
Shader "PoKingHill/Water"
{
    Properties
    {
        _Deep ("Deep colour", Color) = (0.04, 0.20, 0.33, 0.92)
        _Shallow ("Sky / shallow colour", Color) = (0.28, 0.58, 0.72, 0.85)
        _Foam ("Foam colour", Color) = (0.92, 0.96, 1, 1)
        _SeaLevel ("Sea level (m)", Float) = -6
        _PlateauR ("Plateau radius (m)", Float) = 1.5
        _SlopeK ("Slope k", Float) = 0.1111111
        _FoamWidth ("Foam width (m)", Float) = 0.35
        _Storm ("Storm 0..1", Range(0, 1)) = 0
    }
    SubShader
    {
        Tags { "RenderType" = "Transparent" "Queue" = "Transparent" "RenderPipeline" = "UniversalPipeline" }
        Blend SrcAlpha OneMinusSrcAlpha
        ZWrite Off
        Cull Off

        Pass
        {
            Name "Water"
            Tags { "LightMode" = "UniversalForward" }
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_fog
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"

            CBUFFER_START(UnityPerMaterial)
                half4 _Deep, _Shallow, _Foam;
                float _SeaLevel, _PlateauR, _SlopeK, _FoamWidth, _Storm;
            CBUFFER_END

            struct Attributes { float4 positionOS : POSITION; };
            struct Varyings { float4 positionCS : SV_POSITION; float3 positionWS : TEXCOORD0; half fog : TEXCOORD1; };

            Varyings vert(Attributes v)
            {
                Varyings o;
                o.positionWS = TransformObjectToWorld(v.positionOS.xyz);
                o.positionCS = TransformWorldToHClip(o.positionWS);
                o.fog = ComputeFogFactor(o.positionCS.z);
                return o;
            }

            // slope of a sum of moving sines: returns d(height)/dx, d(height)/dz
            float2 Waves(float2 p, float t, float rough)
            {
                float2 g = 0;
                const float2 dirs[4] = { float2(1, 0.3), float2(-0.4, 1), float2(0.7, -0.8), float2(-1, -0.2) };
                const float freq[4] = { 1.3, 2.1, 3.7, 6.1 };
                const float speed[4] = { 0.9, 1.3, 1.9, 2.6 };
                [unroll] for (int i = 0; i < 4; i++)
                {
                    float2 d = normalize(dirs[i]); float ph = dot(d, p) * freq[i] + t * speed[i];
                    g += d * cos(ph) * (0.05 + 0.05 * rough) / (1 + i);
                }
                return g;
            }

            half4 frag(Varyings i) : SV_Target
            {
                float t = _Time.y;
                float2 slope = Waves(i.positionWS.xz, t, _Storm);
                float3 n = normalize(float3(-slope.x, 1, -slope.y));
                float3 view = normalize(GetWorldSpaceViewDir(i.positionWS));
                float fres = pow(1 - saturate(dot(n, view)), 4);
                half3 deep = lerp(_Deep.rgb, _Deep.rgb * 0.55, _Storm);
                half3 col = lerp(deep, _Shallow.rgb * (1 - 0.4 * _Storm), fres);
                Light sun = GetMainLight();
                float3 h = normalize(sun.direction + view);
                col += sun.color * pow(saturate(dot(n, h)), 180) * (1 - 0.7 * _Storm);
                // shoreline: radius where the dome surface reaches the water level
                float r = length(i.positionWS.xz);
                float shore = _SeaLevel >= 0 ? 0 : _PlateauR + sqrt(max(0, -_SeaLevel / _SlopeK));
                float foamBand = saturate(1 - abs(r - shore) / _FoamWidth);
                float foam = foamBand * (0.55 + 0.45 * sin(r * 22 - t * 3 + slope.x * 40));
                col = lerp(col, _Foam.rgb, saturate(foam));
                half alpha = lerp(_Deep.a, 1, max(fres * 0.5, foam));
                col = MixFog(col, i.fog);
                return half4(col, alpha);
            }
            ENDHLSL
        }
    }
    FallBack Off
}
