using UnityEngine;
using UnityEngine.UIElements;

namespace PoKingHill
{
    /// <summary>
    /// A badge's vitals strip, drawn with Painter2D: balance bar (Biometrics stability; green, amber, red), the last 3 s
    /// of joint strain as a sparkline (white, red above 60 % of the force limits). Repainted only when a value changed.
    /// </summary>
    public class VitalsBar : VisualElement
    {
        float _stab = 1f; bool _out; int _head = -1; Biometrics.Vitals _v;
        static readonly Color Good = new(0.43f, 0.84f, 0.47f), Warn = new(1f, 0.77f, 0.25f), Bad = new(0.94f, 0.38f, 0.34f), Track = new(1f, 1f, 1f, 0.14f);

        public VitalsBar() { AddToClassList("vitals"); pickingMode = PickingMode.Ignore; generateVisualContent += Draw; }

        public void Set(Biometrics.Vitals v, bool isOut)
        {
            if (v == _v && v.head == _head && Mathf.Abs(v.stability - _stab) < 0.01f && isOut == _out) return;
            _v = v; _head = v.head; _stab = v.stability; _out = isOut; MarkDirtyRepaint();
        }

        void Draw(MeshGenerationContext ctx)
        {
            var p = ctx.painter2D; var r = contentRect; if (r.width < 4 || _v == null) return;
            float w = r.width, barH = r.height * 0.32f;
            Bar(p, 0, w, barH, Track); if (!_out) Bar(p, 0, w * Mathf.Clamp01(_stab), barH, _stab > 0.6f ? Good : _stab > 0.3f ? Warn : Bad);
            // strain sparkline, oldest sample on the left
            float y0 = barH + r.height * 0.08f, sh = r.height * 0.6f;
            p.lineWidth = 2f; p.lineJoin = LineJoin.Round; p.strokeColor = _out ? Track : new Color(1f, 1f, 1f, 0.8f);
            p.BeginPath(); float peak = 0;
            for (int k = 0; k < Biometrics.History; k++)
            {
                float s = Mathf.Clamp01(_v.strainHistory[(_v.head + k) % Biometrics.History] / 0.8f); peak = Mathf.Max(peak, s);
                var pt = new Vector2(w * k / (Biometrics.History - 1), y0 + sh * (1f - s));
                if (k == 0) p.MoveTo(pt); else p.LineTo(pt);
            }
            if (peak > 0.75f && !_out) p.strokeColor = Bad;
            p.Stroke();
        }

        static void Bar(Painter2D p, float x, float w, float h, Color c)
        {
            if (w <= 0) return;
            p.fillColor = c; p.BeginPath(); p.MoveTo(new Vector2(x, 0)); p.LineTo(new Vector2(x + w, 0)); p.LineTo(new Vector2(x + w, h)); p.LineTo(new Vector2(x, h)); p.ClosePath(); p.Fill();
        }
    }

    /// <summary>Small vector icons drawn with Painter2D, so no font glyph or texture is needed.</summary>
    public class Icon : VisualElement
    {
        public enum Kind { Menu, Sliders, Play }
        readonly Kind _kind;
        public Icon(Kind kind) { _kind = kind; AddToClassList("icon"); pickingMode = PickingMode.Ignore; generateVisualContent += Draw; }

        void Draw(MeshGenerationContext ctx)
        {
            var p = ctx.painter2D; var r = contentRect; float s = Mathf.Min(r.width, r.height); var c = r.center; float u = s * 0.18f;
            p.strokeColor = new Color(0.96f, 0.97f, 0.98f); p.lineWidth = Mathf.Max(2f, s * 0.07f); p.lineCap = LineCap.Round;
            if (_kind == Kind.Menu)
                for (int i = -1; i <= 1; i++) { p.BeginPath(); p.MoveTo(new Vector2(c.x - u * 1.4f, c.y + i * u)); p.LineTo(new Vector2(c.x + u * 1.4f, c.y + i * u)); p.Stroke(); }
            else if (_kind == Kind.Play)
            {
                p.fillColor = new Color(1f, 0.77f, 0.25f); p.BeginPath();
                p.MoveTo(new Vector2(c.x - s * 0.3f, c.y - s * 0.36f)); p.LineTo(new Vector2(c.x + s * 0.38f, c.y)); p.LineTo(new Vector2(c.x - s * 0.3f, c.y + s * 0.36f)); p.ClosePath(); p.Fill();
            }
            else
                for (int i = -1; i <= 1; i += 2)
                {
                    p.BeginPath(); p.MoveTo(new Vector2(c.x - u * 1.4f, c.y + i * u * 0.8f)); p.LineTo(new Vector2(c.x + u * 1.4f, c.y + i * u * 0.8f)); p.Stroke();
                    p.fillColor = new Color(1f, 0.77f, 0.25f); p.BeginPath(); p.Arc(new Vector2(c.x + i * u * 0.6f, c.y + i * u * 0.8f), u * 0.42f, 0, 360); p.Fill();
                }
        }
    }
}
