using System;
using System.Collections.Generic;
using UnityEngine;

namespace PoKingHill
{
    /// <summary>
    /// Drives a skinned character from the MuJoCo bodies of one fighter: every bone of the scan follows one body.
    /// Render only. The character's rest pose is the physics model's zero pose (tools/make_kim.py bakes that into
    /// the mesh), so a bone keeps the offset it has from its body in that pose. The table of bone, body and their
    /// zero-pose positions (MuJoCo coordinates) comes from koth/build_kim.py as kim_skin.json.
    /// </summary>
    public class SkinFollower : MonoBehaviour
    {
        [Serializable] class Row { public string bone, body; public float[] bone_pos, body_pos; }
        [Serializable] class Table { public Row[] bones; }

        public TextAsset skinJson;
        public string robotPrefix = "a_";
        public Transform skinRoot;                       // the instantiated character (its bones are found by name)

        Transform[] _bone, _body; Vector3[] _posOff; Quaternion[] _rotOff;

        static Vector3 Mj(float[] p) => new(p[0], p[2], p[1]);     // MuJoCo x, y, z -> Unity x, z, y

        void Start()
        {
            var rows = JsonUtility.FromJson<Table>(skinJson.text).bones;
            var byName = new Dictionary<string, Transform>();
            foreach (var t in skinRoot.GetComponentsInChildren<Transform>(true)) byName[t.name] = t;
            // The importer may leave the character turned about the vertical: take the quarter turn whose rest pose
            // lands on the physics zero pose.
            skinRoot.position = Vector3.zero; float best = float.MaxValue, bestYaw = 0f;
            for (int k = 0; k < 4; k++)
            {
                skinRoot.rotation = Quaternion.Euler(0f, 90f * k, 0f); float err = 0f;
                foreach (var r in rows) err = Mathf.Max(err, Vector3.Distance(byName[r.bone].position, Mj(r.bone_pos)));
                if (err < best) { best = err; bestYaw = 90f * k; }
            }
            skinRoot.rotation = Quaternion.Euler(0f, bestYaw, 0f);
            Debug.Log($"[SkinFollower] {robotPrefix}: rest pose matches the physics zero pose within {best * 1000f:0.0} mm (yaw {bestYaw})");
            if (best > 0.02f) Debug.LogError($"[SkinFollower] {robotPrefix}: the character's rest pose is not the physics zero pose ({best:0.000} m off)");

            int n = rows.Length; _bone = new Transform[n]; _body = new Transform[n]; _posOff = new Vector3[n]; _rotOff = new Quaternion[n];
            for (int i = 0; i < n; i++)
            {
                _bone[i] = byName[rows[i].bone];
                var go = GameObject.Find(robotPrefix + rows[i].body);
                if (go == null) { Debug.LogError("[SkinFollower] no body " + robotPrefix + rows[i].body); enabled = false; return; }
                _body[i] = go.transform;
                _posOff[i] = _bone[i].position - Mj(rows[i].body_pos);      // body frames are world-aligned in the zero pose
                _rotOff[i] = _bone[i].rotation;
            }
            foreach (var smr in skinRoot.GetComponentsInChildren<SkinnedMeshRenderer>()) smr.updateWhenOffscreen = true;
        }

        void LateUpdate()
        {
            if (_bone == null) return;
            for (int i = 0; i < _bone.Length; i++)       // rows are parents-first, so children overwrite what a parent moved
                _bone[i].SetPositionAndRotation(_body[i].position + _body[i].rotation * _posOff[i], _body[i].rotation * _rotOff[i]);
        }
    }
}
