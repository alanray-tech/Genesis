import pytest
import torch

import genesis as gs


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.gpu])
def test_particle_graph_pipeline_builds_and_runs(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1e-3, substeps=2, pipeline="graph"),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.5, -0.5, 0.0),
            upper_bound=(0.5, 0.5, 1.0),
        ),
        sph_options=gs.options.SPHOptions(
            particle_size=0.04,
            lower_bound=(-0.5, -0.5, 0.0),
            upper_bound=(0.5, 0.5, 1.0),
        ),
        pbd_options=gs.options.PBDOptions(particle_size=0.04),
        show_viewer=show_viewer,
    )
    mpm = scene.add_entity(
        morph=gs.morphs.Box(pos=(-0.15, 0.0, 0.45), size=(0.12, 0.12, 0.12)),
        material=gs.materials.MPM.Elastic(sampler="regular"),
    )
    sph = scene.add_entity(
        morph=gs.morphs.Box(pos=(0.15, 0.0, 0.45), size=(0.12, 0.12, 0.12)),
        material=gs.materials.SPH.Liquid(sampler="regular"),
    )
    pbd = scene.add_entity(
        morph=gs.morphs.Mesh(file="meshes/cloth.obj", pos=(0.0, 0.0, 0.7), scale=0.1),
        material=gs.materials.PBD.Cloth(),
    )
    tool = scene.add_entity(
        morph=gs.morphs.Mesh(file="meshes/stirrer.obj", pos=(0.0, 0.0, 0.45), scale=0.2),
        material=gs.materials.Tool(),
    )
    scene.build()
    for _ in range(3):
        scene.step()

    for entity in (mpm, sph, pbd):
        assert torch.isfinite(entity.get_particles_pos()).all()
    assert torch.isfinite(tool.get_state().pos).all()
