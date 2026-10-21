using UnityEngine;
using UnityEngine.InputSystem;

namespace PoKingHill
{
    /// <summary>
    /// Puppet mode: the player steers one fighter. Its brain is the walker, trained to follow velocity commands, so the
    /// stick sets the command directly: up/down = forward speed (-1..1 m/s), left/right = turn rate (-1..1 rad/s),
    /// Q/E or a second finger's sideways drag = sidestep (-0.5..0.5 m/s), the ranges the walker was trained on.
    /// The HUD's on-screen stick writes <see cref="Stick"/>; the keyboard (WASD / arrows) works too.
    /// </summary>
    public class PuppetControl : MonoBehaviour
    {
        public MatchDirector director;
        /// <summary>x = turn, y = forward, both -1..1. Written by the HUD stick.</summary>
        public static Vector2 Stick;
        public static float Side;
        public float speed = 1f, turn = 1f, sidestep = 0.5f;

        void Update()
        {
            if (director == null || director.PuppetBody < 0 || director.State != MatchState.Fight) return;
            Vector2 s = Stick; float side = Side;
            var k = Keyboard.current;
            if (k != null)
            {
                s.y += (k.wKey.isPressed || k.upArrowKey.isPressed ? 1 : 0) - (k.sKey.isPressed || k.downArrowKey.isPressed ? 1 : 0);
                s.x += (k.dKey.isPressed || k.rightArrowKey.isPressed ? 1 : 0) - (k.aKey.isPressed || k.leftArrowKey.isPressed ? 1 : 0);
                side += (k.qKey.isPressed ? 1 : 0) - (k.eKey.isPressed ? 1 : 0);
            }
            s = Vector2.ClampMagnitude(s, 1f);
            director.bodies[director.PuppetBody].command = new Vector3(speed * s.y, sidestep * Mathf.Clamp(side, -1f, 1f), -turn * s.x);
        }
    }
}
