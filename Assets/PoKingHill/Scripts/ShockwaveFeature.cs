using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.RenderGraphModule.Util;
using UnityEngine.Rendering.Universal;

namespace PoKingHill
{
    /// <summary>
    /// The screen-space ring of a big impact: the image is pushed outward along a ring that grows from the hit for
    /// about 0.45 s (FX/Shockwave.shader). MatchFx starts a wave with Shockwave.Trigger; the pass is only added to the
    /// frame while a wave is running, so it costs nothing the rest of the time. Render only.
    /// </summary>
    public static class Shockwave
    {
        public const float Seconds = 0.45f;
        static Vector3 _world; static float _born = -10f, _strength;
        static readonly int Center = Shader.PropertyToID("_ShockCenter"), Params = Shader.PropertyToID("_ShockParams");

        public static bool Active => Time.unscaledTime - _born < Seconds;

        public static void Trigger(Vector3 world, float strength) { _world = world; _born = Time.unscaledTime; _strength = Mathf.Clamp01(strength); }

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
        static void Reset() => _born = -10f;

        /// <summary>Shader inputs for one camera (globals, so the material asset is never written): centre in viewport uv, then (radius, ring width, strength, aspect).</summary>
        public static void Apply(Camera cam)
        {
            Vector3 v = cam.WorldToViewportPoint(_world);
            float t = Mathf.Clamp01((Time.unscaledTime - _born) / Seconds);
            float fade = v.z > 0 ? (1f - t) * (1f - t) : 0f;
            Shader.SetGlobalVector(Center, new Vector4(v.x, v.y, 0, 0));
            Shader.SetGlobalVector(Params, new Vector4(Mathf.Lerp(0.02f, 0.45f, Mathf.Sqrt(t)), 0.06f, 0.05f * _strength * fade, cam.aspect));
        }
    }

    public class ShockwaveFeature : ScriptableRendererFeature
    {
        public Material material;
        ShockPass _pass;

        public override void Create() { _pass = new ShockPass { renderPassEvent = RenderPassEvent.BeforeRenderingPostProcessing, requiresIntermediateTexture = true }; }

        public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
        {
            // play mode only: in the editor a wave time left from a play session (fast enter-play keeps statics) could still read as running
            if (material == null || !Application.isPlaying || !Shockwave.Active || renderingData.cameraData.cameraType != CameraType.Game || !GameSettings.Effects) return;
            _pass.material = material; _pass.camera = renderingData.cameraData.camera;
            renderer.EnqueuePass(_pass);
        }

        class ShockPass : ScriptableRenderPass
        {
            public Material material; public Camera camera;

            public override void RecordRenderGraph(RenderGraph graph, ContextContainer frameData)
            {
                var res = frameData.Get<UniversalResourceData>();
                if (res.isActiveTargetBackBuffer || material == null) return;
                Shockwave.Apply(camera);
                var src = res.activeColorTexture;
                var desc = graph.GetTextureDesc(src); desc.name = "_ShockwaveColor"; desc.clearBuffer = false;
                var dst = graph.CreateTexture(desc);
                graph.AddBlitPass(new RenderGraphUtils.BlitMaterialParameters(src, dst, material, 0), "Shockwave");
                res.cameraColor = dst;
            }
        }
    }
}
